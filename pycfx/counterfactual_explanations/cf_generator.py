"""
pycfx/counterfactual_explanations/cf_generator.py
Abstract generator for CFXs.
"""

from pycfx.datasets.input_properties import InputProperties
from pycfx.models.abstract_model import AbstractModel
from pycfx.datasets.dim_reduction import DimensionalityReduction

import json
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Type
from pathlib import Path


class CounterfactualGenerator(ABC):
    """
    Abstract generator for CFXs.
    """
    def __init__(self, model: AbstractModel, input_properties: InputProperties, config: dict, save_dir: Path=".", use_pregenerated: bool=True):
        """
        Initialise with `model`, dataset `input_properties` and config dict.
        Set save_dir and use_pregenerated to allow generators to save and reuse data.
        """
        self.model = model
        self.input_properties = input_properties
        self.save_dir = save_dir
        self.config = config
        self.use_pregenerated = use_pregenerated

    @abstractmethod
    def generate_counterfactual(self, x: np.ndarray, y_target: int) -> np.array:
        """
        Generate and return a counterfactual instance for factual instance `x` and target class `y_target`
        """
        pass

    def name(self, exclude=None):
        """
        Get the name of this generator
        """
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Type):
                    return str(obj)
                if isinstance(obj, DimensionalityReduction):
                    return obj.name()
                return super().default(obj)
        
        config = self.config

        if exclude:
            config = {k: v for k, v in config.items() if k != exclude}
    
        return self.__class__.__name__ + json.dumps(config, separators=(',', ':'), cls=CustomEncoder)

    def setup(self, X_train: np.ndarray, y_train: np.ndarray, X_calib: np.ndarray, y_calib: np.ndarray):
        """
        Provide training data (X_train, y_train) and calibration data (X_calib, y_calib) to generators to allow them to set up.
        """
        pass

    def generate_counterfactuals(self, x_factuals: np.ndarray, y_targets: np.ndarray) -> np.array:
        """
        Generate a batch of CFXs for specified `x_factuals` and `y_targets`.
        """

        x_cfs = np.zeros_like(x_factuals)

        if hasattr(self, 'grb_model'):
            sorted_indices = np.lexsort((x_factuals[:, 0], y_targets))
            x_factuals = x_factuals[sorted_indices]
            y_targets = y_targets[sorted_indices]
            self.grb_model.setParam('OutputFlag', 0)


        for i in tqdm(range(x_factuals.shape[0])):
            x_cfs[i] = self.generate_counterfactual(x_factuals[i], y_targets[i])
        
        if hasattr(self, 'grb_model'):
            inverse_indices = np.empty_like(sorted_indices)
            inverse_indices[sorted_indices] = np.arange(len(sorted_indices))
            x_cfs = x_cfs[inverse_indices]
            self.grb_model.setParam('OutputFlag', 1)

        return x_cfs
    
    def check_solution(self, input_mvar: np.ndarray, y_target: np.ndarray, pert_distance: float=0.01) -> np.array:
        """
        Check a pertubation distance `pert_distance` in every dimension to check if (due to numerical issues) the CFX (`input_mvar`) is on the decision boundary but not allocated to the target class `y_target`
        """

        identified_sol = input_mvar.X

        if np.argmax(self.model.predict(identified_sol)) == y_target:
            return identified_sol

        for i in range(len(identified_sol)):
            original_value = identified_sol[i]
            for perturbation in [-pert_distance, pert_distance]:
                identified_sol[i] = original_value + perturbation
                if np.argmax(self.model.predict(identified_sol)) == y_target:
                    return identified_sol
            identified_sol[i] = original_value  

        return identified_sol