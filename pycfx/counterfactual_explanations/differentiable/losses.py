"""
pycfx/counterfactual_explanations/differentiable/losses.py
Losses for use by GradientBasedGenerator
"""

from pycfx.helpers.constants import BACKEND_PYTORCH, BACKEND_TENSORFLOW
from pycfx.counterfactual_explanations.differentiable.optimisation_loop import OptimisationState
from abc import abstractmethod, ABC

import torch
import numpy as np
import tensorflow as tf
from collections import defaultdict

"""
Registry of losses: Dictionary of loss names to implementations, where implementations is a dictionary of backend names to Loss subclass
"""
LOSSES_REGISTRY = defaultdict(dict)

def register_loss(base_cls, backend):
    """
    Decorator to register this class as an implemenation of the loss `base_cls` for backend `backend`
    """
    def decorator(subclass):
        LOSSES_REGISTRY[base_cls][backend] = subclass
        return subclass
    return decorator

def get_loss_with_backend(base_cls, backend_name):
    """
    Get the implemenation of the loss specified by `base_cls` with the backend `backend_name`
    """
    registry_impls = LOSSES_REGISTRY.get(base_cls)

    if registry_impls is None:
        raise ValueError(f"{base_cls.__name__} not found")

    if registry_impls.get(backend_name):
        return registry_impls[backend_name]

    if registry_impls.get(None):
        return registry_impls[None]

    raise NotImplementedError(f"{base_cls.__name__} not implemented on backend '{backend_name}'")


class Loss:
    """Base class that dispatches to the backend-specific Loss subclass."""
    def __new__(cls, *args, **kwargs):
        if cls is Loss:
            raise TypeError("Cannot instantiate Loss directly")

        backend = kwargs.pop("backend", BACKEND_PYTORCH)
        if cls.__bases__[0] is Loss:
            impl_cls = get_loss_with_backend(cls, backend)
            return impl_cls(*args, **kwargs)
        return super().__new__(cls)

    @abstractmethod
    def loss(self, opt_state: OptimisationState):
        """
        Obtain the loss tensor, specifiying an optimisation state `opt_state`
        """
        
        pass


class ClassificationLoss(Loss):
    """
    Get the model's classification loss between the target class and current prediction.
    """

    pass


@register_loss(ClassificationLoss, None)
class _ClassificationLoss(ClassificationLoss):
    def loss(self, opt_state: OptimisationState, **kwargs):
        return opt_state.model.compute_loss(opt_state.y_enc, opt_state.y_target)
    
class DistanceLoss(Loss, ABC):
    """
    Get the distance loss between the factual instance and counterfactual instance.
    Uses the L-`norm` distance, specify `mad` and `mad_data` to use the median absolute deviation, specify `dist_weight` for a custom weighted distance
    """

    def __init__(self, norm: int=1, mad: bool=False, mad_data: np.ndarray=None, dist_weight: np.ndarray=None, **kwargs):
        self.norm = norm
        self.mad = mad
        self.dist_weight = dist_weight

        if self.mad:
            med = np.median(mad_data, axis=0)
            mad = np.median(np.abs(mad_data - med), axis=0)
            mad[mad == 0] = 1e-9 
            self.dist_weight = 1 / mad
        
        
@register_loss(DistanceLoss, BACKEND_PYTORCH)
class _DistanceLossPyTorch(DistanceLoss):
    def __init__(self, norm: int=1, mad: bool=False, mad_data: np.ndarray=None, dist_weight: np.ndarray=None, **kwargs):
        super().__init__(norm, mad, mad_data, dist_weight)

        if self.dist_weight is not None:
            self.dist_weight = torch.tensor(self.dist_weight, dtype=torch.float32)

    def loss(self, opt_state: OptimisationState):
        if self.dist_weight is None:
            return torch.dist(opt_state.z, opt_state.z_factual, p=self.norm)
        else:
            return torch.norm(self.dist_weight.to(opt_state.z.device) * (opt_state.z - opt_state.z_factual), 1)

@register_loss(DistanceLoss, BACKEND_TENSORFLOW)
class _DistanceLossTensorFlow(DistanceLoss):
    def __init__(self, norm: int=1, mad: bool=False, mad_data: np.ndarray=None, dist_weight: np.ndarray=None, **kwargs):
        super().__init__(norm, mad, mad_data, dist_weight)

        if self.dist_weight is not None:
            self.dist_weight = tf.constant(self.dist_weight)

    def loss(self, opt_state: OptimisationState):
        if self.dist_weight is None:
            return tf.norm(opt_state.z - opt_state.z_factual, ord=self.norm)
        else:
            return tf.norm(self.dist_weight * (opt_state.z - opt_state.z_factual), ord=self.norm)
    

class EnergyLoss(Loss):
    """
    Get the (decayed) energy loss for the obtained counterfactual instance
    Used in ECCCo from Altmeyer et al. "Faithful model explanations through energy-constrained conformal counterfactuals." (2024)
    Implementation ported from https://github.com/pat-alt/ECCCo.jl/blob/main/pycfx/penalties.jl
    """

    def __init__(self, reg_strength: float = 1e-3, decay: float = 0.9, **kwargs):
        self.decay = decay
        self.reg_strength = reg_strength

@register_loss(EnergyLoss, BACKEND_PYTORCH)
class _EnergyLossPyTorch(EnergyLoss):
    def loss(self, opt_state: OptimisationState, **kwargs):
        model = opt_state.model.pytorch_model
        x_prime = opt_state.x_enc           
        t = torch.argmax(opt_state.y_target)
        step = opt_state.it + 1
        max_steps = opt_state.n_it

        # polynomial decay multiplier 
        b = round(max_steps / 25)
        a = b / 10
        phi = (1.0 + step / b) ** (-self.decay) * a

        # generative loss (negative logit of target class) 
        logits = torch.softmax(model(x_prime), dim=-1)
        gen_loss = -logits[t]

        # total loss
        if self.reg_strength == 0.0:
            loss = phi * gen_loss
        else:
            reg_loss = torch.norm(gen_loss) ** 2
            loss = phi * (gen_loss + self.reg_strength * reg_loss)

        return loss

@register_loss(EnergyLoss, BACKEND_TENSORFLOW)
class _EnergyLossTensorFlow(EnergyLoss):
    def loss(self, opt_state: OptimisationState, **kwargs):
        model = opt_state.model.keras_model
        x_prime = tf.expand_dims(opt_state.x_enc, 0)
        t = tf.argmax(opt_state.y_target)
        step = opt_state.it + 1
        max_steps = opt_state.n_it

        # polynomial decay multiplier 
        b = round(max_steps / 25)
        a = b / 10
        phi = (1.0 + step / b) ** (-self.decay) * a

        # generative loss (negative logit of target class) 
        logits = tf.nn.softmax(tf.squeeze(model(x_prime), 0), axis=-1)
        gen_loss = -logits[t]

        # total loss
        if self.reg_strength == 0.0:
            loss = phi * gen_loss
        else:
            reg_loss = tf.norm(gen_loss) ** 2
            loss = phi * (gen_loss + self.reg_strength * reg_loss)

        return loss
