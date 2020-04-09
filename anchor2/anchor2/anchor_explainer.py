import logging as logger
from abc import ABC, abstractmethod
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from .anchor_selector import BeamAnchorSearch, GreedyAnchorSearch
from .tabular_explanation import TabularExplanation
from .utils import DiscretizerTransformer


class AnchorExplainer(ABC):

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def explain(self, *args, **kwargs):
        pass


class TabularExplainer(AnchorExplainer):

    def fit(self,
            data: Union[np.array, pd.DataFrame],
            label_decoders: Dict[int, List[str]],
            ordinal_features_idx: List[int],
            oh_encoded_categories: Dict[str, List[int]],
            feature_names: List[str] = None
            ):
        """
        Fits an instance of TabularExplainer with hyperparameters
        :param data: Union[np.array, pd.DataFrame] data which will be used for explaining samples passed
        :param feature_names: List of feature names
        :param label_decoders: Dictionary from feature idx to list of labels used for label encoding for that feature model_name
        :param ordinal_features_idx: Indices of float features
        :param oh_encoded_categories: Dictionary of "Feature model_name" -> List of feature indices in data used for encoding this "feature model_name"
        """
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

        data = np.array(data)
        self.discretizer = DiscretizerTransformer()
        self.discretizer.fit(data, self.ordinal_features_idx, oh_encoded_categories)
        self.data = data
        self.discretized_data = self.discretizer.transform(data)
        self.label_decoders = label_decoders
        # self.translators = self.discretizer.map_translators(label_decoders, self.feature_names)

    def explain(self, x: np.array,
                classifier_fn,
                strategy: str = "kl-lucb",
                threshold=0.95,
                selector_params: Dict = {},
                verbose=True) -> TabularExplanation:

        if strategy == 'kl-lucb':
            logger.info("Using Kullback-Leibler LUCB method")
            selector = BeamAnchorSearch
        elif strategy == "greedy":
            logger.info("Using greedy search method")
            selector = GreedyAnchorSearch

        else:
            raise ValueError("Strategy is not recognized, possible options are ['greedy', 'kl-lucb']")

        explanation: TabularExplanation = selector.find_explanation(x=x,
                                                                    data=self.data,
                                                                    d_data=self.discretized_data,
                                                                    classifier_fn=classifier_fn,
                                                                    d_classifier_fn=lambda y: classifier_fn(
                                                                        self.discretizer.inverse_transform(y)),
                                                                    ordinal_idx=self.ordinal_features_idx,
                                                                    feature_names=self.feature_names,
                                                                    precision_threshold=threshold,
                                                                    **selector_params
                                                                    )

        # translator = ExplanationTranslator()
        # translator.fit(self.translators, self.ordinal_features_idx)
        # explanation.str = translator.transform(explanation)

        return explanation
