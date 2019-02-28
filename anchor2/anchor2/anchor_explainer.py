from abc import ABC, abstractmethod
from typing import Callable, Union, List

import numpy as np
import pandas as pd


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

        # Discretize data

    def explain(self, x: np.array, strategy: str = "kl-lucb", threshold=0.95):
        pass


class TextExplainer(AnchorExplainer):
    pass
