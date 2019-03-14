from abc import ABC, abstractmethod
from typing import Callable, Union, List, Dict

import numpy as np
import pandas as pd

from anchor2.anchor2.anchor_selector import BeamAnchorSearch, GreedyAnchorSearch, AnchorSelectionStrategy
from anchor2.anchor2.utils import DiscretizerTransformer


class AnchorExplainer(ABC):
    pass


class TabularExplainer(AnchorExplainer):

    def fit(self,
            classifier_fn: Callable,
            data: Union[np.array, pd.DataFrame],
            feature_names: List[str],
            ordinal_features_idx: List[int],
            oh_encoded_categories: Dict[str, List[int]]
            ):

        assert len(data.shape) == 2, "Data should be a matrix"

        # Store feature names to generate human-readable explanations
        if feature_names is not None:
            assert len(feature_names) == data.shape[1], "Length of feature names list must match number of columns"
            self.feature_names = feature_names
        elif type(data) is pd.DataFrame:
            self.feature_names = list(data.columns)
        else:
            assert np.issubdtype(data.dtype, np.floating), "Only float np.arrays are accceptable"
            self.feature_names = list(range(data.shape[1]))

        if ordinal_features_idx is not None:
            assert 0 <= min(ordinal_features_idx), "Index of ordinal feature must be positive"
            assert max(ordinal_features_idx) <= data.shape[1], "Ordinal feature index cannot exceed number of columns"
            self.ordinal_features_idx = ordinal_features_idx
        else:
            self.ordinal_features_idx = []

        self.discretizer = DiscretizerTransformer()
        self.discretizer.fit(data, self.ordinal_features_idx, oh_encoded_categories)
        self.data = data
        self.discretized_data = self.discretizer.transform(data)

    def explain(self, x: np.array, classifier_fn, strategy: str = "kl-lucb", threshold=0.95):

        if strategy == 'kl-lucb':
            print("Using Kullback-Leibler LUCB method")
            selector = BeamAnchorSearch
        elif strategy == "greedy":
            selector = GreedyAnchorSearch
            print("Using greedy search method")
        else:
            raise ValueError("Strategy is not recognized, possible options are ['greedy', 'kl-lucb']")

        explanation = selector.find_explanation(x=x,
                                                data=self.data,
                                                classifier_fn=lambda x: classifier_fn(self.discretizer.inverse_transform(x)),
                                                ordinal_idx=self.ordinal_features_idx,
                                                feature_names=self.feature_names,
                                                precision_threshold=threshold,
                                                )
        return explanation


class TextExplainer(AnchorExplainer):
    pass
