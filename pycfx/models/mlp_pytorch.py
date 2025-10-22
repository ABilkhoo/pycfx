"""
pycfx/models/mlp_pytorch.py
Abstract wrapper for any PyTorch model, and a specific PyTorch MLP implementation
"""

from pycfx.models.abstract_model import DifferentiableModel, MILPEncodableModel
from pycfx.datasets.input_properties import InputProperties
from pycfx.helpers.constants import BACKEND_PYTORCH

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, f1_score
import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.torch.sequential import add_sequential_constr
from abc import abstractmethod
from pathlib import Path

class PyTorchModel(DifferentiableModel):
    """
    Abstract wrapper for building, training, saving, loading and evaluating any PyTorch model
    """

    def __init__(self, config: dict, input_properties: InputProperties):
        """
        Initialise with dataset `input_properties` and `config`
        The config key `random_state` is set to a fixed 0, hidden_dims set to [50], batch size to 64, epochs to 50, and learning rate to 0.01
        """
         
        super().__init__(config, input_properties)
        self.batch_size = self.config.get('batch_size', 64)
        self.epochs = self.config.get('epochs', 50)
        self.lr = self.config.get("lr", 0.01)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

        if input_properties is not None:
            self.pytorch_model, self.loss_fn, self.optimiser = self._build_model()
            self.pytorch_model.to(self.device)

    @abstractmethod
    def _build_model(self):
        """
        Use self.config and self.device, and return a tuple of (PyTorch model, PyTorch loss function and PyTorch optimiser)
        """
        pass

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        torch.manual_seed(self.random_state)
        cfg = self.config

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)

        if self.input_properties.y_onehot:
            n_classes = self.input_properties.n_targets
            y_train_onehot = nn.functional.one_hot(y_train_tensor, num_classes=n_classes).float()
            train_dataset = TensorDataset(X_train_tensor, y_train_onehot)
        else:
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.pytorch_model.train()

        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimiser.zero_grad()
                outputs = self.pytorch_model(X_batch)
                loss = self.loss_fn(outputs, y_batch)
                loss.backward()
                self.optimiser.step()

        return self.pytorch_model

    def predict(self, x: np.ndarray) -> np.array:
        return self.pytorch_model(torch.tensor(x, dtype=torch.float).to(self.device)).detach().cpu().numpy()

    def load(self, save_path: Path) -> None:
        self.pytorch_model = torch.load(save_path, weights_only=False)
        self.pytorch_model.to(self.device)
        self.save_path = save_path

    def save(self, save_path: Path):
        torch.save(self.pytorch_model, save_path)
        self.save_path = save_path
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        self.pytorch_model.eval()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(self.device)

        with torch.no_grad():
            test_outputs = self.pytorch_model(X_test_tensor)
            test_loss = self.loss_fn(test_outputs, y_test_tensor).item()
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
            precision = precision_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
            f1 = f1_score(y_test_tensor.cpu().numpy(), predicted.cpu().numpy(), average='weighted')

        model_performance = {
            'loss': test_loss,
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'f1_score': f1 * 100
        }

        return model_performance
    
    def compute_loss(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y1, y2)
    
    def load_external(self, model):
        self.pytorch_model = model
    
    def get_backend(self) -> str:
        return BACKEND_PYTORCH

class PyTorchMLP(PyTorchModel, MILPEncodableModel):
    """
    Multi-layer perceptron using PyTorchModel
    """


    def _build_model(self) -> tuple[nn.Sequential, nn.CrossEntropyLoss, optim.Optimizer]:
        if self.config.get("dims"):
            self.dims = self.config["dims"]
        else:
            self.dims = self.config.get('hidden_dims', [50])
            self.dims = [self.input_properties.n_features] + self.dims + [self.input_properties.n_targets]

        layers = []

        for i in range(len(self.dims) - 1):
            layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            layers.append(nn.ReLU())

        layers = layers[:-1]
        
        model = nn.Sequential(*layers)
        model.to(self.device)

        loss_fn = nn.CrossEntropyLoss()
        optimiser = optim.Adam(model.parameters(), lr=self.lr)

        return model, loss_fn, optimiser
    
    def gp_set_model_constraints(self, grb_model: gp.Model, input_mvar: gp.MVar) -> gp.MVar:
        output_mvar = grb_model.addMVar(shape=(self.pytorch_model[-1].out_features,), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="output")
        add_sequential_constr(grb_model, self.pytorch_model.cpu(), input_mvar, output_mvar)
        self.pytorch_model.to(self.device)
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

