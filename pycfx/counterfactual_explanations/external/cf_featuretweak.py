"""
pycfx/counterfactual_explanations/external/cf_featuretweak.py
Wrapper over existing implementation of FeatureTweak from https://github.com/upura/featureTweakPy/blob/master/featureTweakPy.py
"""

from pycfx.counterfactual_explanations.cf_generator import CounterfactualGenerator
from pycfx.datasets.input_properties import InputProperties
from pycfx.library.featureTweakPy import feature_tweaking
from pycfx.models.randomforest_sklearn import RandomForestSKLearn

import numpy as np
from pathlib import Path

def distance_cost_l1(a, b):
    return np.linalg.norm(a-b, ord=1)

def distance_cost_l2(a, b):
    return np.linalg.norm(a-b, ord=2)

class FeatureTweakGenerator(CounterfactualGenerator):
    """
    Wrapper over existing implementation of FeatureTweak from https://github.com/upura/featureTweakPy/blob/master/featureTweakPy.py
    """

    def __init__(self, model: RandomForestSKLearn, input_properties: InputProperties, config: dict, save_dir:Path =".", use_pregenerated: bool=True):
        """
        Initialise with `model`, dataset `input_properties` and config dict.
        Specify within the config `cost_fn` any distance function between two arrays, two of which can be imported from cf_featuretweak. Additionally an epsilon value, default 0.1.
        """
        super().__init__(model, input_properties, config, save_dir, use_pregenerated)
        assert isinstance(self.model, RandomForestSKLearn)

        self.epsilon = self.config.get('epsilon', 0.1)
        self.cost_fn = self.config.get('cost_fn', distance_cost_l1)
        
    def generate_counterfactual(self, x: np.ndarray, y_target: np.ndarray) -> np.array:
        x_cf = feature_tweaking(self.model.model, x, self.input_properties.get_labels(), y_target, self.epsilon, self.cost_fn)
        return x_cf