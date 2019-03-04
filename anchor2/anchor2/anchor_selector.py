from typing import List
import numpy as np

from anchor2.anchor2.tabular_explanation import TabularExplanation, EqualityPredicate, InequalityPredicate, GreaterOrEqualPredicate, \
    LessPredicate


class AnchorSelectionStrategy:
    pass


class GreedyAnchorSearch(AnchorSelectionStrategy):
    pass


class TabularAnchorGenerator:

    def __init__(self, feature_values, feature_names, ordinal_idx):
        self.feature_values = feature_values
        self.feature_names = feature_names
        self.ordinal_idx = ordinal_idx
        self.anchor_generator = self._get_random_predicate_generator()

    def get_initial_explanations(self, num) -> List[TabularExplanation]:
        explanations = [TabularExplanation()] * num
        return [explanation.increment(self.anchor_generator) for explanation in explanations]

    def _get_random_predicate_generator(self, ):
        """
        :return: Infinite generator of tabular predicates
        """
        feature_idxs = list(range(len(self.feature_values)))
        while True:
            feature_idx = np.random.choice(feature_idxs)
            feature_value_idx = np.random.choice(list(range(len(self.feature_values[feature_idx]))))
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


def compute_metrics(anchor, data, labels, target_label):
    # TODO either this way, or do data augmentation
    vectorized_np_selector = np.vectorize(anchor.numpy_selector())
    data_with_anchor_index = vectorized_np_selector(data)
    data_with_anchor = data[data_with_anchor_index]
    labels_with_anchor = labels[data_with_anchor_index]

    coverage = data.shape[0] / data_with_anchor.shape[0]
    precision = np.mean(labels_with_anchor == target_label)
    return precision, coverage


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
                         ):

        labels = classifier_fn(data)
        target_label = classifier_fn(x)

        feature_values = []
        for i in range(data.shape[1]):
            feature_values.append(np.unique(data[:, i]))

        anchor_generator = TabularAnchorGenerator(feature_values, feature_names, ordinal_idx)
        anchors: List[TabularExplanation] = anchor_generator.get_initial_explanations(anchor_pool_size)
        metrics = np.array([compute_metrics(anchor, data, labels, target_label) for anchor in anchors])

        if np.any(metrics[:, 0] > precision_threshold):
            # Return the anchor with higher coverage
            pass
        else:
            # Launch KL-LUCB and beam search
            pass

        pass
