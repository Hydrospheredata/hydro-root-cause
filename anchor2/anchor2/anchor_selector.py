from typing import List, Tuple
import numpy as np
from copy import deepcopy

from functools import partial
from .explanation import Explanation
from .multi_armed_bandit_solver import BernoulliArm, KullbackLeiblerLUCB
from .tabular_explanation import TabularExplanation, EqualityPredicate, InequalityPredicate, GreaterOrEqualPredicate, \
    LessPredicate
from itertools import compress
from loguru import logger
from jsonschema import validate


class AnchorSelectionStrategy:
    @staticmethod
    def find_explanation(x,
                         data,
                         d_data,
                         classifier_fn,
                         d_classifier_fn,
                         ordinal_idx,
                         feature_names,
                         precision_threshold,
                         anchor_pool_size=15,
                         beam_size=5,
                         batch_size=10,
                         tolerance=0.05,
                         delta=0.1
                         ) -> Explanation:
        pass


class GreedyAnchorSearch(AnchorSelectionStrategy):
    pass


class TabularAnchorFactory:
    def __init__(self, feature_values, feature_names, ordinal_idx):
        self.feature_values = feature_values
        self.feature_names = feature_names
        self.ordinal_idx = ordinal_idx

    def get_initial_explanations(self, sample, num) -> List[TabularExplanation]:
        explanations = [TabularExplanation(sample, self._get_random_predicate_generator(), self.feature_values) for _ in range(num)]
        return [explanation.increment() for explanation in explanations]

    def _get_random_predicate_generator(self, ):
        """
        :return: Infinite generator of random tabular predicates
        """
        while True:
            feature_idx = np.random.randint(low=0, high=len(self.feature_values))
            feature_value_idx = np.random.randint(low=0, high=len(self.feature_values[feature_idx]))
            feature_value = self.feature_values[feature_idx][feature_value_idx]

            if feature_idx in self.ordinal_idx:
                if feature_value_idx == 0:
                    yield GreaterOrEqualPredicate(feature_value, feature_idx, self.feature_names[feature_idx])
                elif feature_value_idx == len(self.feature_values[feature_idx]) - 1:
                    yield LessPredicate(feature_value, feature_idx, self.feature_names[feature_idx])
                else:
                    random_value = np.random.rand()
                    if random_value > 0.66:  # 2/6 chance for GEQ predicate
                        yield GreaterOrEqualPredicate(feature_value, feature_idx, self.feature_names[feature_idx])
                    elif random_value > 0.33:  # 2/6 chance for LE predicate
                        yield LessPredicate(feature_value, feature_idx, self.feature_names[feature_idx])
                    elif random_value > 0.1:  # 23% chance for Equality predicate
                        yield EqualityPredicate(feature_value, feature_idx, self.feature_names[feature_idx])
                    else:  # 10% chance for Inequality predicate
                        yield InequalityPredicate(feature_value, feature_idx, self.feature_names[feature_idx])
            else:
                if np.random.rand() > 0.01:  # Why 0.25?
                    yield EqualityPredicate(feature_value, feature_idx, self.feature_names[feature_idx])
                else:
                    yield InequalityPredicate(feature_value, feature_idx, self.feature_names[feature_idx])


def compute_metrics_on_original_data(anchor, d_data, labels, target_label) -> Tuple[float, float]:
    """
    Compute metrics on a subset of data for which anchor applies
    :param anchor: Anchor for which metrics are computed
    :param d_data: data with ordinal features binned, and values replaced with corresponding bin number
    :param labels: Labels produced by the explained classifier
    :param target_label: the label for the explained sample
    :return: (precision, coverage)
    """

    data_with_anchor_index = np.apply_along_axis(anchor.check_against_sample, axis=1, arr=d_data)
    data_with_anchor = d_data[data_with_anchor_index]
    labels_with_anchor = labels[data_with_anchor_index]

    coverage: float = data_with_anchor.shape[0] / d_data.shape[0]
    precision: float = np.mean(labels_with_anchor == target_label)

    anchor._coverages.append(coverage)
    anchor._precisions.append(precision)

    logger.debug(f"{anchor} \t\t with pr. {precision:.3f} and cov. {coverage:.3f} ({data_with_anchor.shape[0]}/{d_data.shape[0]}).")

    return precision, coverage


def compute_reward_on_augmented_data(anchor: TabularExplanation,
                                     d_data_batch: np.array,
                                     d_classifier_fn,
                                     target_label,
                                     ):
    """
    Creates a copy of data and alters it's rows, so all of them do satisfy anchor.
    :param anchor: Explanation, set of predicates
    :param d_data_batch: Discretized data
    :param d_classifier_fn: Classifier function which maps discretized data to continuous and returns labels
    :param target_label: label of the sample we try to define
    :return: Reward. Number of samples classified the same as target_label, after each sample was changed to satisfy anchor
    """
    data_copy = d_data_batch.copy()
    for feature_id in range(d_data_batch.shape[1]):
        feature_masks = [
            np.ones((d_data_batch.shape[0], 1))]  # Boolean mask with 1 - predicate is true for the sample, 0 - predicate is false
        feature_values = d_data_batch[:, feature_id]  # Feature column
        for predicate in filter(lambda p: p.feature_id == feature_id, anchor.predicates):
            # Mask is generated for each predicate and masks are stacked
            feature_masks.append(predicate.check_against_column(feature_values).reshape(-1, 1))
        feature_mask = np.all(np.concatenate(feature_masks, axis=1), axis=1)  # Masks are merged
        assert feature_mask.shape[0] == d_data_batch.shape[0]

        if np.sum(feature_mask) < d_data_batch.shape[0]:
            # Values which do not satisfy predicates, are replaced with possible values
            possible_values = anchor.get_possible_feature_values(feature_id)

            # FIXME This line sometimes throw "ValueError: possible_values must be non-empty"
            possible_values = np.random.choice(possible_values, size=np.sum(~feature_mask), replace=True)

            data_copy[~feature_mask, feature_id] = possible_values

    reward = np.sum(d_classifier_fn(d_data_batch) == target_label)
    return reward


def compute_reward_on_batch(d_data, batch_size, anchor, d_classifier_fn, target_label):
    """
    Computes a classifier prediction for a subsample of data. Subsample of data is selected
    according to an anchor. If no such subsample can be acquired - data is augmented, and
    "reward", or mean label is calculated on augmented data

    :param d_data: discretized data np.array from which batch is selected
    :param batch_size: Batch size
    :param anchor: Explanation for which reward should be computed
    :param d_classifier_fn: Explained classifier function, but for discretized domain. Basically, this is classifier_fn, applied to
    discretizer.inverse_transform(d_data).
    :param target_label: label of explained sample
    :return: (batch_size, cumulative_reward)
    """
    subsample_indices = np.random.choice(list(range(d_data.shape[0])), size=batch_size, replace=False)
    subsampled_data = d_data[subsample_indices]
    return batch_size, compute_reward_on_augmented_data(anchor, subsampled_data, d_classifier_fn, target_label)


class BeamAnchorSearch(AnchorSelectionStrategy):

    @staticmethod
    def find_explanation(x,
                         data,
                         d_data,
                         classifier_fn,
                         d_classifier_fn,
                         ordinal_idx: List[int],
                         feature_names: List[str],
                         precision_threshold: float = 0.95,
                         anchor_pool_size=25,
                         beam_size=5,
                         batch_size=150,
                         tolerance=0.3,
                         delta=0.2,
                         **kwargs
                         ) -> Explanation:
        """
        This function returns the first anchor which precision will be >= precision_threshold. Anchor selection process is an iterative
         process - first, the pool of anchors is created, then best anchors anchors from this pool take over the whole pool by copying
         themselves.
        :param x: Explained sample
        :param data: Original data
        :param d_data: Discretized version of data
        :param classifier_fn: Explained function
        :param d_classifier_fn: Explained function which works on discretized domain
        :param ordinal_idx: Indices of ordinal features
        :param feature_names: List of feature names
        :param precision_threshold: the minimum precision sufficient for  returning an anchor
        :param anchor_pool_size: The # of anchors evaluated at each step of the selection process
        :param beam_size: # of anchors selected at each step of the selection process
        :param batch_size: # of samples evaluated at request of selector algorithm (kl-lucb)
        :param tolerance: minimum distance between lowest boundary of best-n arms
        and highest upper boundary of other arms to be achieved by an algorithm. Stopping
        criteria for an algorithm
        :param delta: Mistake probability. It is the probability that chosen set of arms
        will not be a subset of optimal arms.
        :return: Anchor object
        """
        labels = classifier_fn(data)
        target_label = classifier_fn(x.reshape(1, -1))

        feature_values = []
        for i in range(d_data.shape[1]):
            feature_values.append(np.unique(d_data[:, i]))

        anchor_generator = TabularAnchorFactory(feature_values, feature_names, ordinal_idx)
        anchors: List[TabularExplanation] = anchor_generator.get_initial_explanations(x, anchor_pool_size)

        # metrics[:, 0] is a precision, metrics[:, 1] is a coverage
        metrics = np.array([compute_metrics_on_original_data(anchor, data, labels, target_label) for anchor in anchors])

        # Compute mean precision for debugging purposes
        mean_precision, mean_coverage = np.mean(metrics, axis=0)
        logger.info(f"Mean precision == {mean_precision:.3f}")

        while not np.any(metrics[:, 0] > precision_threshold):

            draw_fns = [partial(compute_reward_on_batch,
                                d_data=d_data,
                                batch_size=batch_size,
                                anchor=a,
                                d_classifier_fn=d_classifier_fn,
                                target_label=target_label) for a in anchors]

            # Represent each anchor as a Bernoulli distribution with mean equal to anchor precision
            arms = [BernoulliArm(anchor, draw_fn) for anchor, draw_fn in zip(anchors, draw_fns)]

            # Use KullbackLeiblerLUCB algorithm which selects subset of best bernoulli distributions
            arm_selector = KullbackLeiblerLUCB(arms)
            best_arms = arm_selector.get(beam_size, delta, tolerance)

            # Clone best_arms to fill the whole list of anchors pool.
            # Idea - p as a function of coverage?
            best_arms_duplication_indices = np.random.choice(list(range(len(best_arms))), size=anchor_pool_size, replace=True)

            new_anchors = [best_arms[i].obj.copy() for i in best_arms_duplication_indices]
            anchors.clear()  # Clear the old pool
            anchors = new_anchors  # Change pool to clones of best_anchors

            for anchor in anchors:
                anchor.increment()  # Increment each anchor in the pool by 1 predicate

            # Compute precision and coverage for each anchor
            metrics = np.array([compute_metrics_on_original_data(anchor, data, labels, target_label) for anchor in anchors])

            # Compute mean precision for debugging purposes
            mean_precision, mean_coverage = np.mean(metrics, axis=0)
            logger.info(f"Mean precision == {mean_precision:.3f}")

        # Select best anchor by maximum coverage among all anchors which satisfy precision threshold
        satisfactory_anchors_mask = metrics[:, 0] >= precision_threshold
        satisfactory_anchors = list(compress(anchors, satisfactory_anchors_mask))
        metrics = metrics[satisfactory_anchors_mask]

        logger.info("-----" * 30)
        logger.debug("Satisfactory anchors:")
        for a, (p, c) in zip(satisfactory_anchors, metrics):
            logger.debug(f"{a} \t with pr. {p:.3f} and cov. {c:.3f}.")
        logger.debug("-----" * 30)

        best_anchor = satisfactory_anchors[metrics[:, 1].argmax()]
        logger.info("Best anchor: " + str(best_anchor))

        return best_anchor
