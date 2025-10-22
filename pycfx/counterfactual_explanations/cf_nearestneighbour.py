"""
pycfx/counterfactual_explanations/cf_nearestneighbour.py
Nearest Neighbour Counterfactual Explanation
"""

from pycfx.models.abstract_model import AbstractModel
from pycfx.counterfactual_explanations.cf_generator import CounterfactualGenerator
from pycfx.datasets.input_properties import InputProperties

import numpy as np
from pathlib import Path


class NearestNeighbourCF(CounterfactualGenerator):
    """
    NearestNeighbourCF: Return the nearest training point of the target class to the factual instance as the CFX.
    """
    def __init__(self, model: AbstractModel, input_properties: InputProperties, config: dict, save_dir: Path=".", use_pregenerated: bool=True):
        """
        Initialise with `model`, dataset `input_properties` and config dict.
        Set save_dir and use_pregenerated to allow generators to save and reuse data.
        Alter config "ord" to specify the norm used in the distance computation.
        """
        super().__init__(model, input_properties, config, save_dir, use_pregenerated)
        self.ord = self.config.get('ord', 2)
    
    def setup(self, X_train: np.ndarray, y_train: np.ndarray, X_calib: np.ndarray, y_calib: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.predictions = np.argmax(self.model.predict(X_train), axis=1)

    def generate_counterfactual(self, x: np.ndarray, y_target: int) -> np.array:
        target_class_points = self.X_train[np.where(self.y_train == y_target) and np.where(self.predictions == y_target)]
        distances = np.linalg.norm(target_class_points - x, ord=self.ord, axis=1)
        min_dist_index = np.argmin(distances)
        return target_class_points[min_dist_index]

        