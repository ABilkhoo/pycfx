"""
pycfx/counterfactual_explanations/differentiable/optimisation_loop.py
Optimisation loops to allow a CFX to be obtained by minimising specified losses and using gradient descent.
"""

from pycfx.models.latent_encodings import IdentityEncoding
from pycfx.models.abstract_model import DifferentiableModel
from pycfx.helpers.constants import BACKEND_PYTORCH

import numpy as np
from abc import ABC, abstractmethod

"""
Registry of optimisation loops: dictionary of backend to implementation
"""
OPTIMISATION_LOOPS = {}

def register_optimisation_loop(backend: str):
    """
    Decorator to add an optimisation loop implementation for backend `backend` to registry
    """
    def decorator(subclass):
        assert issubclass(subclass, DifferentiableOptimisation)
        OPTIMISATION_LOOPS[backend] = subclass
        return subclass
    return decorator


class DifferentiableOptimisation(ABC):
    """
    Abstract class to define an optimisation loop for generating CFXs.
    Dispatches to the backend-specific optimisation loop, according to the model.
    """

    def __new__(cls, *args, **kwargs):
        backend = args[0].get_backend()

        if cls is DifferentiableOptimisation and OPTIMISATION_LOOPS.get(backend):
            impl_cls = OPTIMISATION_LOOPS[backend]
            return impl_cls(*args, **kwargs)

        return super().__new__(cls)
    
    def __init__(self, model, input_properties, losses, n_iter, lr, min_max_lambda=None, losses_weights=None, latent_encoding=IdentityEncoding(), jsma=False, early_stopping=False, retain_graph=False, **kwargs):
        """
        Initialise an optimisation loop with `model`, dataset `input_properties`, list of Loss instances `losses`, max number of iterations `n_iter`,
        learning rate for optimiser `lr`, `min_max_lambda` for min-max optimisation, weights for losses `losses_weights`, latent encoding `latent_encoding`, whether to use salient feature optimiser `jsma`,
        whether to stop early if no change is detected in the CFX `early_stopping`, and whether to retain the graph in backwards autograd (Torch-specific)
        """
        self.model = model
        self.input_properties = input_properties
        self.losses = losses
        self.n_iter = n_iter
        self.lr = lr
        self.early_stopping = early_stopping
        self.min_max_lambda = min_max_lambda
        self.losses_weights = losses_weights
        self.latent_encoding = latent_encoding
        self.retain_graph=retain_graph
        if self.losses_weights is None and self.losses is not None:
            self.losses_weights = np.ones(len(self.losses))
            assert len(self.losses_weights) == len(self.losses), "losses and losses_weight length mismatch"

        self.jsma = jsma
        self.tensor_bounds = None

    @abstractmethod
    def optimise_minmax(self, x: np.ndarray, y_target: int) -> np.array:
        """
        Find a CFX for the given factual `x` and target `y_target`, by maximising the element of the loss function and minimising all others, governed by the `min_max_lambda` specified at initialisation.
        """
        pass

    def optimise_min(self, x: np.ndarray, y_target: int) -> np.array:
        """
        Find a CFX for the given factual `x` and target `y_target`, by minimising the loss functions specified at initialisation.
        """
        pass

class OptimisationState:
    """
    Dataclass for the optimisation state.
    """

    def __init__(self, model: DifferentiableModel, z, z_factual, x_enc, y_enc, x_factual, y_factual, y_target, it: int, n_it: int):
        self.model = model
        self.z = z
        self.z_factual = z_factual
        self.x_enc = x_enc
        self.y_enc = y_enc
        self.x_factual = x_factual
        self.y_factual = y_factual
        self.y_target = y_target
        self.it = it
        self.n_it = n_it

    def __str__(self):
        return (f"OptimisationState(model={self.model}, z={self.z}, z_factual={self.z_factual}, "
                f"x_enc={self.x_enc}, y_enc={self.y_enc}, x_factual={self.x_factual}, "
                f"y_factual={self.y_factual}, y_target={self.y_target}, it={self.it}, n_it={self.n_it})")