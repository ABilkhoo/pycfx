"""
pycfx/counterfactual_explanations/cf_gradient_based.py
Gradient based CFX generators.
"""

from pycfx.datasets.input_properties import InputProperties
from pycfx.counterfactual_explanations.differentiable.losses import DistanceLoss, ClassificationLoss, EnergyLoss
from pycfx.counterfactual_explanations.differentiable.optimisation_loop import DifferentiableOptimisation
from pycfx.counterfactual_explanations.cf_generator import CounterfactualGenerator
from pycfx.conformal.localised_conformal_lcp import SplitConformalPrediction
from pycfx.conformal.losses_conformal import SetSizeLoss
from pycfx.models.abstract_model import DifferentiableModel
from pycfx.models.variational_autoencoder import VariationalAutoencoder

import numpy as np
from pathlib import Path

class GradientBasedGenerator(CounterfactualGenerator):
    """
    Abstract gradient based generator
    """
    def __init__(self, model: DifferentiableModel, input_properties: InputProperties, config: dict = {}, save_dir: Path=None, use_pregenerated: bool=True):
        """
        Initialise with `model`, dataset `input_properties` and config dict.
        Set config's `norm`, `dist_weight`, `mad` to specify distance computations.
        Set `n_iter`, learning rate `lr`, and `min_max_lambda` (for Wachter) as optimisation loop parameters.

        Set save_dir and use_pregenerated to allow generators to save and reuse data.
        """
        super().__init__(model, input_properties, config, save_dir, use_pregenerated)

        assert isinstance(self.model, DifferentiableModel)
        self.model = model

        self.norm = self.config.get('norm', 1)
        self.dist_weight = self.config.get('dist_weight', None)
        self.mad = self.config.get('mad', False)
        self.n_iter = self.config.get('n_iter', 1000)
        self.lr = self.config.get('lr', 0.005)
        self.min_max_lambda = self.config.get('min_max_lambda', 1)
        self.backend = self.model.get_backend()

class WachterGenerator(GradientBasedGenerator):
    """
    Wachter et. al. "Counterfactual explanations without opening the black box: Automated decisions and the GDPR." (2017)
    """

    def setup(self, X_train: np.ndarray, y_train: np.ndarray, X_calib: np.ndarray, y_calib: np.ndarray):
        distance_loss = DistanceLoss(self.norm, self.mad, X_train, backend=self.backend)
        clf_loss = ClassificationLoss(backend=self.backend)

        self.optimisation_loop = DifferentiableOptimisation(self.model, self.input_properties, losses=[clf_loss, distance_loss], n_iter=self.n_iter, lr=self.lr, min_max_lambda=self.min_max_lambda, early_stopping=True)

    def generate_counterfactual(self, x: np.ndarray, y_target: int):
        return self.optimisation_loop.optimise_minmax(x, y_target)

class ECCCOGenerator(GradientBasedGenerator):
    """
    ECCCo: Altmeyer et al. "Faithful model explanations through energy-constrained conformal counterfactuals." (2024)
    """

    
    def setup(self, X_train: np.ndarray, y_train: np.ndarray, X_calib: np.ndarray, y_calib: np.ndarray):
        clf_loss = ClassificationLoss(backend=self.backend)
        distance_loss = DistanceLoss(self.norm, self.mad, X_train, backend=self.backend)
        energy_based_loss = EnergyLoss(backend=self.backend)

        conformal_config = self.config.get('conformal_config', {})
        conformal = SplitConformalPrediction(self.model, self.input_properties, conformal_config, self.save_dir, self.use_pregenerated)
        conformal.calibrate(X_calib, y_calib)

        set_size_loss = SetSizeLoss(conformal, backend=self.backend)
        
        self.optimisation_loop = DifferentiableOptimisation(self.model, self.input_properties, losses=[clf_loss, distance_loss, energy_based_loss, set_size_loss], n_iter=self.n_iter, lr=self.lr, 
                                                                  losses_weights=[1, 0.2, 0.4, 0.4])

    def generate_counterfactual(self, x: np.ndarray, y_target: int):
        return self.optimisation_loop.optimise_min(x, y_target)

class SchutGenerator(GradientBasedGenerator):
    """
    Schut et al. "Generating interpretable counterfactual explanations by implicit minimisation of epistemic and aleatoric uncertainties." (2021).
    Note: Schut takes an ensemble of models.
    """
    def setup(self, X_train: np.ndarray, y_train: np.ndarray, X_calib: np.ndarray, y_calib: np.ndarray):
        clf_loss = ClassificationLoss(backend=self.backend)
        self.optimisation_loop = DifferentiableOptimisation(self.model, self.input_properties, losses=[clf_loss], n_iter=self.n_iter, lr=self.lr, min_max_lambda=None, jsma=True)

    def generate_counterfactual(self, x: np.ndarray, y_target: int):
        return self.optimisation_loop.optimise_min(x, y_target)
    

class ReviseGenerator(GradientBasedGenerator):
    """
    REVISE: Joshi et al. "Towards realistic individual recourse and actionable explanations in black-box decision making systems." (2019).
    """
    # Distance + loss + latent VAE
    def setup(self, X_train: np.ndarray, y_train: np.ndarray, X_calib: np.ndarray, y_calib: np.ndarray):
        distance_loss = DistanceLoss(self.norm, self.mad, X_train, backend=self.backend)
        clf_loss = ClassificationLoss(backend=self.backend)

        vae_config = self.config.get("vae_config", {})
        #Train VAE, save
        encoder = VariationalAutoencoder(vae_config, self.input_properties)

        if self.save_dir is not None:
            encoder.load_or_train(self.save_dir / "vae", X_train, y_train, self.use_pregenerated) 
        else:
            encoder.train(X_train, y_train) 

        # self.encoder = encoder

        self.optimisation_loop = DifferentiableOptimisation(self.model, self.input_properties, losses=[clf_loss, distance_loss], n_iter=self.n_iter, lr=self.lr, min_max_lambda=self.min_max_lambda, latent_encoding=encoder,
                                                                  losses_weights=[1, 1])

    def generate_counterfactual(self, x: np.ndarray, y_target: int):
        return self.optimisation_loop.optimise_min(x, y_target)