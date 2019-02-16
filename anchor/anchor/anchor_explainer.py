import numpy as _np

from typing import Callable as _Callable
from .anchor_explanation import AnchorExplanation as _AnchorExplanation
from .anchor_tabular import AnchorTabularExplainer as _AnchorTabularExplainer


class AnchorExplainer:
    """
    Facade class to AnchorTabularExplainer, AnchorTextExplainer, etc
    """

    def __init__(self, mode='tabular', **kwargs):
        """
        Initialize class with appropriate explainer based on provided mode and kwargs.
        :param mode: One of ("tabular", "image", "text")
        :param kwargs: kwargs for AnchorTabularExplainer, AnchorTextExplainer, AnchorImageExplainer, etc.
        """
        if mode != 'tabular':
            raise NotImplementedError("Only tabular mode is implemented")

        self.mode = mode

        # These parameters will be provided during self.fit()
        self.data = None
        self.target = None
        self.prediction_fn = None

        self.__dict__.update(kwargs)  # kwargs are hyperparameters specified for particular for mode

        if mode == 'tabular':
            self.explainer = _AnchorTabularExplainer()

            if 'categorical_features_idx' not in self.__dict__:
                raise ValueError("Please, provide 'categorical_features_idx' argument")
            if 'categorical_names' not in self.__dict__:
                raise ValueError("Please, provide 'categorical_names' argument")
            if 'class_names' not in self.__dict__:
                raise ValueError("Please, provide 'class_names' argument")
            if 'balance' not in self.__dict__:
                raise ValueError("Please, provide 'balance' argument")

    def fit(self, prediction_fn: _Callable, data: _np.array, target: _np.array) -> 'AnchorExplainer':
        """
        Fit explainer with function which needs to be predicted and data.
        :param prediction_fn: function which needs to be interpreted
        :param data: data on which prediction_fn operates
        :param target: labels predicted for data
        :return: self
        """
        self.data = data
        self.target = target
        self.prediction_fn = prediction_fn
        self.explainer.fit(data, target, **self.__dict__)
        return self

    def explain(self, ) -> _AnchorExplanation:
        pass
