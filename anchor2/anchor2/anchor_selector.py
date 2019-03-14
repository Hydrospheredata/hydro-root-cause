from typing import List, Tuple
import numpy as np
from copy import deepcopy

from functools import partial
from anchor2.anchor2.explanation import Explanation
from anchor2.anchor2.multi_armed_bandit_solver import BernoulliArm, KullbackLeiblerLUCB
from anchor2.anchor2.tabular_explanation import TabularExplanation, EqualityPredicate, InequalityPredicate, GreaterOrEqualPredicate, \
    LessPredicate


class AnchorSelectionStrategy:
    @staticmethod
    def find_explanation(x,
                         data,
                         classifier_fn,
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


class TabularAnchorGenerator:
    def __init__(self, feature_values, feature_names, ordinal_idx):
        self.feature_values = feature_values
        self.feature_names = feature_names
        self.ordinal_idx = ordinal_idx
        self.anchor_generator = self._get_random_predicate_generator()

    def get_initial_explanations(self, sample, num) -> List[TabularExplanation]:
        explanations = [TabularExplanation(sample, self.anchor_generator)] * num
        return [explanation.increment() for explanation in explanations]

    def _get_random_predicate_generator(self, ):
        """
        :return: Infinite generator of tabular predicates
        """
        feature_indices = list(range(len(self.feature_values)))
        while True:
            feature_idx = np.random.choice(feature_indices, size=1)
            feature_value_idx = np.random.choice(list(range(len(self.feature_values[feature_idx]))), size=1)
            feature_value = self.feature_values[feature_idx]

            if feature_idx in self.ordinal_idx:
                if feature_value_idx == 0:
                    yield GreaterOrEqualPredicate(feature_value, feature_idx, self.feature_names[feature_idx])
                elif feature_value_idx == len(self.feature_values[feature_idx]) - 1:
                    yield LessPredicate(feature_value, feature_idx, self.feature_names[feature_idx])
                else:
                    if np.random.rand() > 0.5:  # We may introduce some function of distance between idx and [0, max_len] instead of 0.5
                        yield GreaterOrEqualPredicate(feature_value, feature_idx, self.feature_names[feature_idx])
                    else:
                        yield LessPredicate(feature_value, feature_idx, self.feature_names[feature_idx])
            else:
                if np.random.rand() > 0.25:  # Why 0.25?
                    yield EqualityPredicate(feature_value, feature_idx, self.feature_names[feature_idx])
                else:
                    yield InequalityPredicate(feature_value, feature_idx, self.feature_names[feature_idx])


def compute_metrics_on_original_data(anchor, discretized_data, labels, target_label) -> Tuple[float, float]:
    """
    Compute metrics on a subset of data selected by predicates in the anchor
    :param anchor: Anchor for which metrics are computed
    :param discretized_data: data with ordinal features binned, and values replaced with corresponding bin number
    :param labels: Labels produced by the explained classifier
    :param target_label: the label for the explained sample
    :return: (precision, coverage)
    """
    vectorized_np_selector = np.vectorize(anchor.numpy_selector())
    data_with_anchor_index = vectorized_np_selector(discretized_data)
    data_with_anchor = discretized_data[data_with_anchor_index]
    labels_with_anchor = labels[data_with_anchor_index]

    coverage = discretized_data.shape[0] / data_with_anchor.shape[0]
    precision = np.mean(labels_with_anchor == target_label)
    return precision, coverage


def compute_reward_on_augmented_data(anchor: TabularExplanation, data: np.array, classifier_fn, target_label):
    """
    Creates a copy of data and alters it's rows, so all of them do satisfy anchor.
    :param anchor: Explanation, set of predicates
    :param data: Discretized data
    :param classifier_fn: Classifier function which maps discretized data to continuous and returns labels
    :param target_label: label of the sample we try to define
    :return: Reward. Number of samples classified the same as target_label, after each sample was changed to satisfy anchor
    """
    data_copy = data.copy()
    for feature_id in range(data.shape[1]):
        feature_masks = [np.ones(data.shape[0])]  # Boolean mask with 1 - predicate is true for the sample, 0 - predicate is false
        feature_values = data[:, feature_id]  # Feature column
        for predicate in filter(lambda p: p.feature_id == feature_id, anchor.predicates):
            # Mask is generated for each predicate and masks are stacked
            feature_masks.append(predicate.check_against_column(feature_values))
        feature_mask = np.all(np.concatenate(feature_masks, axis=1), axis=1)  # Masks are merged

        # Values which do not satisfy predicates, are replaced with possible values
        data_copy[~feature_mask, feature_id] = np.random.choice(set(feature_values[feature_mask]), size=np.sum(~feature_mask), replace=True)

    reward = np.sum(classifier_fn(data) == target_label)
    return reward


def compute_reward_on_batch(data, batch_size, anchor, classifier_fn, target_label):
    """
    Function passed to the BernoulliArm constructor, which is used in KL_LUCB arms selector.
    All of it's arguments  re curried before passing to the BernoulliArm(...).
    :param data: np.array from which batch is selected
    :param batch_size: Batch size
    :param anchor: Explanation for which reward should be computed
    :param classifier_fn: Explained classifier function
    :param target_label: label of explained sample
    :return: (batch_size, cumulative_reward)
    """
    subsample_indices = np.random.choice(list(range(data.shape[0])), size=batch_size, replace=False)
    subsampled_data = data[subsample_indices]
    return batch_size, compute_reward_on_augmented_data(anchor, subsampled_data, classifier_fn, target_label)


class BeamAnchorSearch(AnchorSelectionStrategy):

    @staticmethod
    def find_explanation(x,
                         data,
                         classifier_fn,
                         ordinal_idx,
                         feature_names,
                         precision_threshold,
                         anchor_pool_size=15,
                         beam_size=5,
                         batch_size=10,
                         tolerance=0.05,
                         delta=0.1
                         ) -> Explanation:

        labels = classifier_fn(data)
        target_label = classifier_fn(x)

        feature_values = []
        for i in range(data.shape[1]):
            feature_values.append(np.unique(data[:, i]))

        anchor_generator = TabularAnchorGenerator(feature_values, feature_names, ordinal_idx)
        anchors: List[TabularExplanation] = anchor_generator.get_initial_explanations(anchor_pool_size)

        # metrics[:, 0] is a precision, metrics[:, 1] is a coverage
        metrics = np.array([compute_metrics_on_original_data(anchor, data, labels, target_label) for anchor in anchors])

        while np.all(metrics[:, 0] < precision_threshold):
            draw_fns = [partial(compute_reward_on_batch, data=data, batch_size=batch_size,
                                anchor=a, classifier_fn=classifier_fn, target_label=target_label) for a in anchors]
            arms = [BernoulliArm(anchor, draw_fn) for anchor, draw_fn in zip(anchors, draw_fns)]
            arm_selector = KullbackLeiblerLUCB(arms)
            best_arms = arm_selector.get(beam_size, delta, tolerance)

            # Idea - p as a function of coverage?
            best_arms_duplication_indices = np.random.choice(list(range(len(best_arms))), size=anchor_pool_size, replace=True)
            new_anchors_list = [deepcopy(best_arms[i]) for i in best_arms_duplication_indices]
            anchors.clear()
            anchors = new_anchors_list
            metrics = np.array([compute_metrics_on_original_data(anchor, data, labels, target_label) for anchor in anchors])

        # Select best anchor by maximum coverage among all anchors which satisfy precision threshold
        best_anchor = anchors[metrics[metrics[:, 0] < precision_threshold][:, 1].argmax()]
        return best_anchor
