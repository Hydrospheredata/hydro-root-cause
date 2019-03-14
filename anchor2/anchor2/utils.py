from typing import List, Dict

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from itertools import chain


class DiscretizerTransformer:
    def __init__(self):
        """
        This transformer discretizes all ordinal columns and decodes all columns encoded with a one hot encoding
        """
        pass

    def fit(self, data, ordinal_indices: List[int], oh_encoded_categories: Dict[str, List[int]]):
        """
        :param data: pd.Dataframe or np.array
        :param ordinal_indices: Indices of columns with ordinal data
        :param oh_encoded_categories: Map from column name to list of oh-encoded columns
        :return: self
        """
        self.ordinal_indices: List[int] = ordinal_indices
        self.oh_encoded_categories: Dict[str, List[int]] = oh_encoded_categories
        self._number_of_features = data.shape[1]

        # n_bins for each feature is selected by Freedman–Diaconis rule
        iq_ranges = np.array([np.subtract(*np.percentile(data[:, f_idx], [75, 25])) for f_idx in ordinal_indices])
        bins_widths = 2 * iq_ranges / np.cbrt(data.shape[0])
        features_n_bins = np.ceil(np.ptp(data[:, ordinal_indices], axis=0) / bins_widths).astype(np.int)  # FIXME deletion by 0.0
        features_n_bins = np.maximum(features_n_bins, 2)

        discretizers = [KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile") for n_bins in features_n_bins]
        for discretizer, feature_idx in zip(discretizers, ordinal_indices):
            discretizer.fit(data[:, feature_idx].reshape(-1, 1))
        self.discretizers = discretizers

        self.output_number_of_features = data.shape[1] - np.sum([len(x) - 1 for x in oh_encoded_categories.values()])

        self._feature_mapping = {}
        oh_encoded_categories_inverse = dict(chain(*[[(v, k) for v in vs] for k, vs in oh_encoded_categories.items()]))

        oh_encoded_columns = sum(oh_encoded_categories.values(), [])  # list of all indices corresponding to one-hot encoded features
        processed_columns = []  # List to keep track of processed features
        processed_oh_features = []  # List to keep track of processed features

        current_feature_counter = 0
        inverse_feature_mapping = {}  # Feature mapping from new feature_index to original feature_index. One to one or one to many

        for feature_id in range(data.shape[1]):
            # Iterate over every feature
            if feature_id in oh_encoded_columns:
                if feature_id not in processed_columns:
                    if oh_encoded_categories_inverse[feature_id] in processed_oh_features:
                        processed_columns.append(feature_id)
                    else:
                        category = oh_encoded_categories_inverse[feature_id]
                        processed_columns.extend(oh_encoded_categories[category])
                        inverse_feature_mapping[current_feature_counter] = oh_encoded_categories[category]
            else:
                inverse_feature_mapping[current_feature_counter] = feature_id
                processed_columns.append(feature_id)
            current_feature_counter += 1

        self._feature_mapping = dict(
            [(tuple(v) if type(v) is list else v, k) for k, v in inverse_feature_mapping.items()])  # One or many to one mapping

        return self

    def transform(self, data):
        assert data.shape[1] == self._number_of_features, f"Data was fit on {self._number_of_features} features." \
            f" Passed: {data.shape[1]} features"

        # Create empty array, in which we will fill discretized data
        discretized_data = np.zeros((data.shape[0], self.output_number_of_features))

        # Compute discretized version of each ordinal column
        for discretizer, feature_idx in zip(self.discretizers, self.ordinal_indices):
            discretized_data[:, self._feature_mapping[feature_idx]] = discretizer.transform(data[:, feature_idx].reshape(-1, 1))[:, 0]

        # Decode OH columns
        for feature_name, feature_idxs in self.oh_encoded_categories.items():
            assert np.all(data[:, feature_idxs].sum(axis=1) <= 1), "Multi-label one-hot encoding is currently not supported"

            # Following line assumes that first column was dropped during oh encoding.
            oh_decoded_labels = data[:, feature_idxs].argmax(axis=1) + data[:, feature_idxs].sum(axis=1)
            discretized_data[:, self._feature_mapping[tuple(feature_idxs)]] = oh_decoded_labels

        return discretized_data

    def inverse_transform(self, discretized_data):
        """
        Lossy inverse transformation! Can create non-existing data points.
        :param discretized_data: discretized data
        :return:
        """

        # Create empty array, in which we will fill restored data
        restored_data = np.zeros((discretized_data.shape[0], self._number_of_features))

        # Compute discretized version of each ordinal column
        for discretizer, feature_idx in zip(self.discretizers, self.ordinal_indices):
            restored_data[:, feature_idx] = discretizer.inverse_transform(
                discretized_data[:, self._feature_mapping[feature_idx]].reshape(-1, 1))[:, 0]

        # Decode OH columns
        for feature_name, feature_idxs in self.oh_encoded_categories.items():
            # Following line assumes that first column was dropped during oh encoding.
            # print(self._feature_mapping[tuple(feature_idxs)])
            oh_encoded_category = OneHotEncoder().fit_transform(
                discretized_data[:, self._feature_mapping[tuple(feature_idxs)]].reshape(-1, 1))

            restored_data[:, feature_idxs] = oh_encoded_category.toarray()

        return restored_data
