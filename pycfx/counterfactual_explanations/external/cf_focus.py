"""
pycfx/counterfactual_explanations/external/cf_focus.py
Wrapper over existing implementation of FOCUS, from the CFXplorer library. See https://github.com/kyosek/CFXplorer
"""

from pycfx.counterfactual_explanations.cf_generator import CounterfactualGenerator
from pycfx.datasets.input_properties import InputProperties
from pycfx.models.randomforest_sklearn import RandomForestSKLearn

from cfxplorer import Focus
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path

class FOCUSGenerator(CounterfactualGenerator):
    """
    Wrapper over existing implementation of FOCUS, from the CFXplorer library. See https://github.com/kyosek/CFXplorer
    """

    def __init__(self, model: RandomForestSKLearn, input_properties: InputProperties, config: dict, save_dir: Path=".", use_pregenerated: bool=True):
        """
        Initialise with `model`, dataset `input_properties` and config dict.
        Specify within the config `distance_func` from "euclidean", "cosine", "l1" and "mahalabobis". Also specifiy `n_iter`, default 100.
        """
        super().__init__(model, input_properties, config, save_dir, use_pregenerated)
        assert isinstance(self.model, RandomForestSKLearn)
        self.distance_func = self.config.get('distance_func', 'l1')
        self.n_iter = self.config.get('n_iter', 100)
        
    def generate_counterfactual(self, x: np.ndarray, y_target: np.ndarray) -> np.array:
        self.focus = Focus(distance_function=self.distance_func, num_iter=self.n_iter, optimizer=keras.optimizers.Adam(), verbose=0)
        x_cf = self.focus.generate(self.model.model, x.reshape(1, -1).astype(float))[0]
        return x_cf

    def generate_counterfactuals(self, x_factuals: np.ndarray, y_targets: np.ndarray) -> np.array:
        self.focus = Focus(distance_function=self.distance_func, num_iter=self.n_iter, optimizer=keras.optimizers.Adam(), verbose=0)
        x_cf = self.focus.generate(self.model.model, x_factuals.astype(float))
        return x_cf