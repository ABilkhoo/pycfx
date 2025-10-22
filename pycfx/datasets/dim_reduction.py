"""
pycfx/datasets/dim_reduction.py
Dimensionality Reduction, intended for use with the localised conformal prediction methods
"""

from pycfx.datasets.input_properties import InputProperties
from pycfx.models.mlp_pytorch import PyTorchMLP
from pycfx.models.abstract_model import AbstractModel

import torch
from torch import nn, optim
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from abc import ABC
from pathlib import Path

class DimensionalityReduction(ABC):
    """
    Abstract DimensionalityReduction instance
    """
    def __init__(self, target_dim: int):
        """
        Specify `target_dim`, dimension of the space in which points are encoded
        """
        self.target_dim = target_dim

    def setup(self, model: AbstractModel, input_properties: InputProperties, X_train: np.ndarray, y_train: np.ndarray, save_dir: Path=None, use_pretrained: bool=True):
        """
        Setup the dim reduction instance by providing a `model`, dataset `input_properties`, training data (X_train, y_train).
        Set save_dir and use_pretrained to allow this instance to save and re-use computation.
        """
        self.model = model
        self.input_properties = input_properties
        self.X_train = X_train
        self.y_train = y_train
        self.save_dir = save_dir
        self.use_pretrained = use_pretrained

    def encode(self, X: np.ndarray) -> np.array:
        """
        Encode x into the reduced dimensional space
        """
        pass

    def decode(self, X: np.ndarray) -> np.array:
        """
        (Optional) Reverse the encoding of an encoded X
        """
        pass

    def name(self) -> str:
        """
        Obtain the name of the dimensionality reduction method
        """
        return f"{self.__class__.__name__}-target{self.target_dim}"

    def gp_dim_encoding(self, grb_model: gp.Model, X: gp.MVar) -> gp.MVar:
        """
        Given a `grb_model` and an MVar `X`, return a new MVar constrained to the encoding of `X`
        """
        pass

    def gp_dim_decoding(self, grb_model: gp.Model, X: gp.MVar) -> gp.MVar:
        """
        (Optional) Given a `grb_model` and an MVar `X`, return a new MVar constrained to the decoding of `X`
        """

        pass


class PCADimReduction(DimensionalityReduction):
    """
    PCA dim encoding
    """

    def setup(self, model: AbstractModel, input_properties: InputProperties, X_train: np.ndarray, y_train: np.ndarray, save_dir: Path=None, use_pretrained: bool=True):
        super().setup(model, input_properties, X_train, y_train, save_dir, use_pretrained)
        self.pca = PCA(n_components=self.target_dim)
        self.pca.fit(X_train)

    def encode(self, X: np.ndarray) -> np.array:
        return self.pca.transform(X)
    
    def decode(self, X: np.ndarray) -> np.array:
        return self.pca.inverse_transform(X)

    def gp_dim_encoding(self, grb_model: gp.Model, X: gp.MVar):
        y = grb_model.addMVar(shape=(self.target_dim,), lb=-GRB.INFINITY)
        grb_model.addConstr(y == (X - np.reshape(self.pca.mean_, (1, -1))) @ self.pca.components_.T )
        return y

    def gp_dim_decoding(self, grb_model: gp.Model, X: gp.MVar):
        y = grb_model.addMVar(shape=(self.input_properties.n_features,), lb=-GRB.INFINITY)
        grb_model.addConstr(y == X @ self.pca.components_ + self.pca.mean_)
        return y


class LDADimReduction(DimensionalityReduction):
    """
    LDA dim encoding
    """

    def setup(self, model: AbstractModel, input_properties: InputProperties, X_train: np.ndarray, y_train: np.ndarray, save_dir: Path=None, use_pretrained: bool=True):
        super().setup(model, input_properties, X_train, y_train, save_dir, use_pretrained)
        self.lda = LinearDiscriminantAnalysis(n_components=self.target_dim)
        self.lda.fit(X_train, y_train)

    def encode(self, X: np.ndarray) -> np.array:
        return self.lda.transform(X)

    def gp_dim_encoding(self, grb_model: gp.Model, X: gp.MVar):
        #For solver = svd
        y = grb_model.addMVar(shape=(self.target_dim,), lb=-GRB.INFINITY)
        grb_model.addConstr(y == (X - self.lda.xbar_) @ self.lda.scalings_)
        return y
    
class SecondLastLayerReduction(DimensionalityReduction):
    """
    Use the outputs second-last layer of the trained `model` provided in .setup() as the encoding. E.g. see https://arxiv.org/pdf/2206.13092
    """
     
    def setup(self, model: AbstractModel, input_properties: InputProperties, X_train: np.ndarray, y_train: np.ndarray, save_dir: Path=None, use_pretrained: bool=True):
        super().setup(model, input_properties, X_train, y_train, save_dir, use_pretrained)
        self.model = model
        self.target_dim = self.model.config['dims'][-2]

    def encode(self, X: np.ndarray) -> np.array:
        return self.model.model[0:-1](torch.tensor(X, dtype=torch.float)).detach().numpy()

    def gp_dim_encoding(self, grb_model: gp.Model, X: gp.MVar):
        # print(grb_model.getVars())
        layer_sl = list(self.model.model)[-2]
        layer_sl_idx = len(self.model.config['dims']) - 1
        layer_sl_name = layer_sl.__class__.__name__
        size_layer_sl = self.model.config['dims'][-2]

        var_names = [layer_sl_name.lower() + f"_{layer_sl_idx}.act[0,{i}]" for i in range(size_layer_sl)]
        
        if grb_model.getVarByName(var_names[0]) is None:
            var_names = [layer_sl_name.lower() + f"_{layer_sl_idx - 1}.act[0,{i}]" for i in range(size_layer_sl)]

        output_var = grb_model.addMVar(shape=(size_layer_sl,))
        for i in range(output_var.shape[0]):
            grb_model.addConstr(output_var[i] == grb_model.getVarByName(var_names[i]))

        return output_var
    
class AutoencoderDimReduction(DimensionalityReduction):
    """
    Use an autoencoder for dimensionality reduction. Best for handling mixed features.
    """
     

    def setup(self, model: AbstractModel, input_properties: InputProperties, X_train: np.ndarray, y_train: np.ndarray, save_dir: Path=None, use_pretrained: bool=True):
        super().setup(model, input_properties, X_train, y_train, save_dir, use_pretrained)
        input_dim = self.input_properties.n_features
        target_dim = self.target_dim

        class AutoEncoder(nn.Module):
            def __init__(self):
                super(AutoEncoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Linear(input_dim // 2, target_dim),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(target_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Linear(input_dim // 2, input_dim),
                )

            def forward(self, x):
                x = self.encoder(x)
                x = self.decoder(x)
                return x

        ae = AutoEncoder()

        if self.use_pretrained and self.save_dir:
            try:
                ae.load_state_dict(torch.load(f"{self.save_dir}/autoencoder.pth"))
                print("Loaded pretrained autoencoder.")
            except FileNotFoundError:
                print("Pretrained autoencoder not found. Training a new one.")
        
        if not self.use_pretrained or not self.save_dir or not hasattr(ae, 'encoder'):
            loss = nn.MSELoss()
            opt = optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)

            epochs = 5

            torch.manual_seed(2)
            torch.use_deterministic_algorithms(True)

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=400, shuffle=True)

            for epoch in range(epochs):
                for x in train_loader:
                    x = x[0]
                    reconstructed = ae(x)
                    loss_val = loss(reconstructed, x)

                    opt.zero_grad()
                    loss_val.backward()
                    opt.step()

                print(f"Epoch {epoch+1}/{epochs}")

            if self.save_dir:
                torch.save(ae.state_dict(), f"{self.save_dir}/autoencoder.pth")
                print("Saved trained autoencoder.")

        self.ae = ae

    def encode(self, X: np.ndarray) -> np.array:
        return self.ae.encoder(torch.tensor(X, dtype=torch.float)).detach().cpu().numpy()

    def decode(self, X: np.ndarray) -> np.array:
        return self.ae.decoder(torch.tensor(X, dtype=torch.float)).detach().cpu().numpy()

    def gp_dim_encoding(self, grb_model: gp.Model, X: gp.MVar):
        encoder = PyTorchMLP(config={}, input_properties=None)
        encoder.load_external(self.ae.encoder)
        return encoder.gp_set_model_constraints(grb_model, X)

    def gp_dim_decoding(self, grb_model: gp.Model, X: gp.MVar):
        encoder = PyTorchMLP(config={}, input_properties=None)
        encoder.load_external(self.ae.decoder)
        return encoder.gp_set_model_constraints(grb_model, X)
