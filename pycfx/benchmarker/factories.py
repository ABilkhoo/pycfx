"""
pycfx/benchmarker/factories.py
Factories for generating models, generators under various configurations and repeats for use by CFBenchmarker
"""

from pycfx.models.abstract_model import AbstractModel
from pycfx.datasets.input_properties import InputProperties
from pycfx.counterfactual_explanations.cf_generator import CounterfactualGenerator

import os
import json
from collections import defaultdict
from itertools import product
import numpy as np
from typing import Type, List, Dict
from pathlib import Path


class ModelFactory:
    """
    ModelFactory: Create and train models with various configurations over multiple repeats.
    """

    def __init__(self, Model: Type[AbstractModel], input_properties: InputProperties, config: dict, config_multi: dict):
        """
        Create a factory for model class `Model`, with input_properties `input_properties`, specifying the config for all models with `config` and config to make variants with `config_multi`.
        """
        self.Model = Model
        self.input_properties = input_properties
        self.config = config
        self.config_multi = config_multi

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, n_repeats: int, save_dir: Path, use_pretrained: bool=True):
        """
        Train and return `n_repeat * len(product(config_multi.values()))` using the specified train set (X_train, y_train).
        Set save_dir and use_pretrained to allow for saving and loading models.
        """
        os.makedirs(save_dir, exist_ok=True)
        models = []
        models_over_repeat = defaultdict(list)
        random_state = 0

        for vals in product(*self.config_multi.values()):
            config = dict(zip(self.config_multi.keys(), vals))
            config = config | self.config 
            config_str = json.dumps(config, separators=(',', ':'))
            model_path_name = save_dir / f"{self.Model.__name__}_{config_str}"

            for repeat in range(n_repeats):
                repeat_path = model_path_name / f"repeat{repeat}"
                model = self.Model(config | {"random_state": random_state}, self.input_properties)
                model.load_or_train(repeat_path, X_train, y_train, use_pretrained)
                models_over_repeat[model_path_name].append(model)
                models.append(model)
                
                random_state += 1

        self.models = models
        self.models_over_repeat = models_over_repeat

        return models
        
    def get_models(self) -> List[AbstractModel]:
        """
        Get a list all models created by the factory
        """
        return self.models
    
    def get_models_over_repeats(self) -> Dict[str, List[AbstractModel]]:
        """
        Get a dictionary where the keys represent an individual model config, and the values contain the `n_repeat` models for the corresponding config.
        """
        return self.models_over_repeat


class GeneratorFactory:
    """
    ModelFactory: Create and initialise multiple generators with various configurations.
    """

    def __init__(self, generators_classes: List[Type[CounterfactualGenerator]], config: dict, config_multi: dict):
        """
        Create a factory for many generator classes `generators_classes`, specifying the config for all generators with `config` and config to make variants with `config_multi`.
        """
        
        self.generators_classes = generators_classes
        self.config = config
        self.config_multi = config_multi
        self.generators = []

        for config_key, config_val in self.config_multi.items():
            if isinstance(config_val, dict):
                self.config_multi[config_key] = [dict(zip(config_val.keys(), val)) for val in product(*config_val.values())]

    def setup_generators(self, model: AbstractModel, input_properties: InputProperties, X_train: np.ndarray, y_train: np.ndarray, X_calib: np.ndarray, y_calib: np.ndarray, save_dir: Path, use_pretrained: bool=True) -> List[CounterfactualGenerator]:
        """
        Setup and return all generators using the model, input_properties, training data (X_train, y_train), and calibration data (X_calib, y_calib).
        Specify save_dir and use_pretrained to allow generators to store and reuse data during setup.
        """

        generators = []

        for vals in product(*self.config_multi.values()):
            config = dict(zip(self.config_multi.keys(), vals))
            config = config | self.config 
            
            for generator_cls in self.generators_classes:
                generator = generator_cls(model, input_properties, config, save_dir, use_pretrained)
                generator.setup(X_train, y_train, X_calib, y_calib)
                generators.append(generator)

        self.generators = generators
        return self.generators

    def get_generators(self) -> List[CounterfactualGenerator]:
        """
        Get a list of all generators. Call `setup_generators` first.
        """
        return self.generators