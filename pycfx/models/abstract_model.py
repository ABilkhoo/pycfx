"""
pycfx/models/abstract_model.py
Abstract wrapper over a predictive model to allow a CFXs to be produced for a wide array of models with varying interfaces.
"""

from pycfx.datasets.input_properties import InputProperties

import os 
import json
import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import precision_score, f1_score
import gurobipy as gp
from joblib import dump, load
from pathlib import Path

class AbstractModel(ABC):
    """
    Abstract wrapper over a predictive model to allow a CFXs to be produced for a wide array of models with varying interfaces.
    """

    def __init__(self, config: dict, input_properties: InputProperties):
        """
        Initialise with dataset `input_properties` and `config`, which can be used by subclasses to set model hyperparameters. 
        Notably the `random_state` config is set to a fixed seed of 0
        """
        self.model = None
        self.config = config
        self.config["random_state"] = self.config.get("random_state", 0)
        self.random_state = self.config["random_state"]
        self.input_properties = input_properties
        self.save_dir = None

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the model with training data (X_train and y_train)
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.array:
        """
        Get the model prediction for a single example or a batch of examples
        """
        pass

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Get a dictionary containing the model performance over a test set (X_test, y_test), containing by default, accuracy, precision, and F1 score.
        """

        accuracy = self.model.score(X_test, y_test) * 100
        y_pred = self.model.predict(X_test)
        precision = precision_score(y_test, y_pred, average='weighted') * 100
        f1 = f1_score(y_test, y_pred, average='weighted') * 100

        model_performance = {
            'accuracy': accuracy,
            'precision': precision,
            'f1_score': f1
        }

        return model_performance
    
    def load_or_save_evaluation(self, X_test: np.ndarray, y_test: np.ndarray, use_pretrained: bool=True) -> dict: 
        """
        Helper to return a saved evaluation of the model on test set (X_test, y_test), or run the evaluation and save it to model evaluation path if needed.
        Setting `use_pretrained` to False will always recompute new evaluation scores. 
        """

        evaluation_path = self.save_dir / "eval.json"
        if evaluation_path.is_file() and use_pretrained:
            with open(evaluation_path, 'r') as f:
                return json.load(f)
            
        evaluation = self.evaluate(X_test, y_test)
        with open(evaluation_path, 'w') as f:
            json.dump(evaluation, f)
        
        return evaluation

    def savename(self) -> str:
        """
        Returns the model's filename. By default uses a .model extension
        """

        return "saved.model"

    def save(self, save_path: Path) -> None:
        """
        Save the model to `save_path`
        """
        dump(self.model, save_path)

    def load(self, save_path: Path) -> None:
        """
        Load the model from `save_path`
        """
        self.model = load(save_path)
    
    def save_to_dir(self, save_dir: Path) -> None:
        """
        Save the model to the directory `save_dir`, as a file with name given by `self.savename()`
        """

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        model_path = self.save_dir / self.savename()
        self.save(model_path)

        config_path = self.save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f)

    def load_from_dir(self, save_dir: Path) -> None:
        """
        Load the model from the directory `save_dir`, which would be stored as a file with name given by `self.savename()`
        """

        self.save_dir = save_dir
        model_path = self.save_dir / self.savename()
        self.load(model_path)

        config_path = self.save_dir / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def load_external(self, model):
        """
        Provide a `model` object for this wrapper to use. Use to create an AbstractModel wrapper over a model trained externally from this library
        """

        self.model = model

    def load_or_train(self, save_dir: Path, X_train: np.ndarray, y_train: np.ndarray, use_pretrained: bool=True):
        """
        Helper to return a saved trained model on training set (X_train, y_train), or train the model and save to `save_dir` needed.
        Setting `use_pretrained` to False will always re-train and re-save the model.
        """

        model_path = save_dir / self.savename()
        config_path = save_dir / "config.json"
        if model_path.is_file() and config_path.is_file() and use_pretrained:
            self.load_from_dir(save_dir)
            return self.model
        else:
            self.train(X_train, y_train)
            self.save_to_dir(save_dir)
            return self.model

    
class DifferentiableModel(AbstractModel):
    @abstractmethod
    def get_backend() -> str:
        """
        Obtain a string representing the backend of the model. E.g. pytorch or tensorflow. See helpers/constants.py for currently implemented backends.
        Used to choose correct loss objects and optimisation loops.
        """

        pass

    @abstractmethod
    def compute_loss(self, y1: np.ndarray, y2: np.ndarray) -> str:
        """
        Obtain model's classification loss between two tensors of size (n_classes,).
        """
        pass

class MILPEncodableModel(AbstractModel):
    @abstractmethod 
    def gp_set_model_constraints(self, grb_model: gp.Model, input_mvar: gp.MVar) -> gp.MVar:
        """
        Given a `grb_model` and input variables `input_mvar`, return a new MVar which is constrained to be the model output for that input. Often makes use of methods from the gurobi-machinelearning library.
        """
        pass

    @abstractmethod
    def gp_set_classification_constraint(self, grb_model: gp.Model, output_vars: gp.MVar, target_class: int, db_distance=1e-6) -> None:
        """
        Given a `grb_model`, input variables `input_mvar`, output variables `output_mvars`, and a target class, constrain the output_mvars to represent the target class (i.e. model predicts target class). 
        """
        pass