"""
pycfx/models/mlp_keras.py
Abstract wrapper for any keras-backed model, and a specific Keras MLP implementation
"""

from pycfx.models.abstract_model import DifferentiableModel, MILPEncodableModel
from pycfx.datasets.input_properties import InputProperties
from pycfx.helpers.constants import BACKEND_TENSORFLOW

import keras
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.keras import add_keras_constr
from abc import abstractmethod
import tensorflow as tf
import numpy as np
from pathlib import Path

class KerasModel(DifferentiableModel):
    """
    Abstract wrapper for building, training, saving, loading and evaluating any keras-backed model
    """
    def __init__(self, config: dict, input_properties: InputProperties):
        """
        Initialise with dataset `input_properties` and `config`
        The config key `random_state` is set to a fixed 0, hidden_dims set to [50], batch size to 64, epochs to 50, and learning rate to 0.001
        """
        super().__init__(config, input_properties)
        self.batch_size = self.config.get('batch_size', 64)
        self.epochs = self.config.get('epochs', 50)
        self.lr = self.config.get('lr', 0.001)

        if input_properties is not None:
            self.keras_model, self.loss_fn, self.optimiser = self._build_model()

            self.keras_model.compile(
                optimizer=self.optimiser,
                loss=self.loss_fn,
                metrics=[keras.metrics.CategoricalAccuracy()],
            )

    def savename(self) -> str:
        return "saved.keras"

    @abstractmethod
    def _build_model(self):
        """
        Use self.config and return a tuple of (Keras model, Keras loss function and Keras optimiser)
        """
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        if self.input_properties.y_onehot:
            y_train = tf.one_hot(y_train, depth=self.input_properties.n_features)

        self.keras_model.fit(x=X_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs)
        return self.keras_model

    def predict(self, X: np.ndarray) -> np.array:
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            return self.keras_model(X).numpy()[0]
        return self.keras_model(X).numpy()

    def load(self, save_path: Path) -> None:
        self.keras_model = keras.saving.load_model(save_path)
        self.save_path = save_path

    def save(self, save_path: Path) -> None:
        self.keras_model.save(save_path)
        self.save_path = save_path

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        if self.input_properties.y_onehot:
            y_test = tf.one_hot(y_test, depth=self.input_properties.n_features)

        res = self.keras_model.evaluate(X_test, y_test, batch_size=self.batch_size)
        
        return dict(zip(self.keras_model.metrics_names, res))
    
    def compute_loss(self, y1: tf.Tensor, y2: tf.Tensor) -> tf.Tensor:
        return self.loss_fn(y1, y2)
    
    def load_external(self, model):
        self.keras_model = model   

    def get_backend(self) -> str:
        return BACKEND_TENSORFLOW
            

class KerasMLP(KerasModel, MILPEncodableModel):
    """
    Multi-layer perceptron using KerasModel
    """

    def _build_model(self) -> tuple[keras.Sequential, keras.losses.CategoricalCrossentropy, keras.optimizers.Adam]:
        if self.config.get("dims"):
            self.dims = self.config["dims"]
        else:
            self.dims = self.config.get('hidden_dims', [50])
            self.dims = [self.input_properties.n_features] + self.dims + [self.input_properties.n_targets]

        layers = []

        for i in range(len(self.dims) - 1):
            layers.append(keras.layers.Dense(self.dims[i+1], input_shape=(self.dims[i],)))
            layers.append(keras.layers.ReLU())

        layers = layers[:-1]
        
        model = keras.Sequential(layers)

        loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        optimiser = keras.optimizers.Adam(learning_rate=self.lr)

        return model, loss_fn, optimiser
    
    def gp_set_model_constraints(self, grb_model: gp.Model, input_mvar: gp.MVar) -> gp.MVar:
        output_shape = self.keras_model.compute_output_shape(tf.constant([0] * self.input_properties.n_features).shape)
        output_mvar = grb_model.addMVar(shape=output_shape, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="output")
        add_keras_constr(grb_model, self.keras_model, input_mvar, output_mvar)
        return output_mvar
    
    def gp_set_classification_constraint(self, grb_model: gp.Model, output_vars: gp.MVar, target_class: int, db_distance=1e-6) -> None:
        classification_constrs = []

        if output_vars.shape[0] == 1:
            #Single output
            assert target_class in [0, 1], "Target class must be 0 or 1 for a single logit output"

            if target_class == 0:
                c1 = grb_model.addConstr(output_vars[0] >= 0.5, name="Output class 0 constraint")
                classification_constrs.append(c1)
            else:
                c2 = grb_model.addConstr(output_vars[0] <= 0.5, name="Output class 1 constraint")
                classification_constrs.append(c2)
            
        else:
            #One-hot output
            for i in range(output_vars.shape[0]):
                if i != target_class:
                    c = grb_model.addConstr(output_vars[target_class] >= output_vars[i] + db_distance)
                    classification_constrs.append(c)

        
        return classification_constrs
    
