from abc import ABC, abstractmethod
from typing import Callable, Union, List

import numpy as np
import pandas as pd

from anchor2.anchor2.anchor_selector import BeamAnchorSearch, GreedyAnchorSearch, AnchorSelectionStrategy


class AnchorExplainer(ABC):
    pass


class TabularExplainer(AnchorExplainer):

    def fit(self,
            classifier_fn: Callable,
            data: Union[np.array, pd.DataFrame],
            feature_names: List[str],
            ordinal_features_idx: List[int]):

        assert len(data.shape) == 2, "Data should be a matrix"

        # Store feature names to generate human-readable explanations
        if feature_names is not None:
            assert len(feature_names) == data.shape[1], "Length of feature names list must match number of columns"
            self.feature_names = feature_names
        elif type(data) is pd.DataFrame:
            self.feature_names = list(data.columns)
        else:
            self.feature_names = list(range(data.shape[1]))

        if ordinal_features_idx is not None:
            assert 0 <= min(ordinal_features_idx), "Index of ordinal feature must be positive"
            assert max(ordinal_features_idx) <= data.shape[1], "Ordinal feature index cannot exceed number of columns"
            self.ordinal_features_idx = ordinal_features_idx
        elif type(data) is pd.DataFrame:
            # TODO Extract indices of all numerical columns
            pass
        elif type(data) is np.array:
            # TODO Check numpy data types
            pass
        else:
            self.ordinal_features_idx = []

        # TODO Discretize data and save both versions, as well as bin edges.

    def explain(self, x: np.array, strategy: str = "kl-lucb", threshold=0.95):

        selector: AnchorSelectionStrategy = None
        if strategy == 'kl-lucb':
            print("Using Kullback-Leibler LUCB method")
            selector = BeamAnchorSearch
        elif strategy == "greedy":
            selector = GreedyAnchorSearch
            print("Using greedy search method")
        else:
            raise ValueError("Strategy is not recognized, possible options are ['greedy', 'kl-lucb']")

        explanation = selector.find_explanation(x, self.data, threshold=threshold)


class TextExplainer(AnchorExplainer):
    pass
