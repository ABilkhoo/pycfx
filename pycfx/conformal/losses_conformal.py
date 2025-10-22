"""
pycfx/conformal/losses_conformal.py
Set size losses for split conformal prediction and localised conformal prediction.
"""

from pycfx.conformal.split_conformal import SplitConformalPrediction
from pycfx.conformal.localised_conformal_lcp import BaseLCP
from pycfx.counterfactual_explanations.differentiable.losses import Loss, OptimisationState
from pycfx.counterfactual_explanations.differentiable.losses import register_loss, BACKEND_PYTORCH, BACKEND_TENSORFLOW

import torch
import tensorflow as tf
import numpy as np

class SetSizeLoss(Loss):
    """
    SetSizeLoss: Implementation of smooth set size loss from Stutz et. al. "Learning optimal conformal classifiers." (2021).
                 Used in ECCCo from Altmeyer, et al. "Faithful model explanations through energy-constrained conformal counterfactuals." (2024)
    """

    def __init__(self, conformal: SplitConformalPrediction, T=0.4, kappa=1, **kwargs):
        """
        Initialise with a SplitConformalPrediction object, T: temperature for sigmoid fn in soft label assignment, kappa: target set size.
        """
        self.conformal = conformal
        self.T = T
        self.kappa = kappa

    def soft_label_assignment(self, fx, y):
        """Obtain the soft label assignment sigmoid(quantile_val - scorefn(fx, y) / T)"""
        pass

    def loss(self, opt_state: OptimisationState):
        """Obtain the smooth conformal set size for opt_state.x_enc: max(0, sum(soft_label_assignment) - kappa)"""
        pass

@register_loss(SetSizeLoss, BACKEND_PYTORCH)
class SetSizeLoss_PyTorch(SetSizeLoss):
    """PyTorch implementation of SetSizeLoss"""

    def __init__(self, conformal: SplitConformalPrediction, T=0.4, kappa=1, **kwargs):
        super().__init__(conformal, T, kappa)
        self.kappa = torch.tensor(self.kappa)

    def soft_label_assignment(self, fx, y):
        return torch.sigmoid((self.conformal.quantile_val - self.conformal.scorefn(fx, y) )/self.T)

    def loss(self, opt_state: OptimisationState):
        y_enc = opt_state.y_enc
        set_size = torch.tensor(0.0, requires_grad=True)
        
        for label in self.conformal.input_properties.get_labels():
            sla = self.soft_label_assignment(y_enc, label)
            set_size = set_size + sla
            
        return torch.max(torch.tensor(0.0), set_size - self.kappa)            

@register_loss(SetSizeLoss, BACKEND_TENSORFLOW)
class SetSizeLoss_TensorFlow(SetSizeLoss):
    """TensorFlow implementation of SetSizeLoss"""
    def __init__(self, conformal: SplitConformalPrediction, T=0.4, kappa=1, **kwargs):
        super().__init__(conformal, T, kappa)
        self.kappa = tf.constant(self.kappa, dtype=tf.float32)

    def soft_label_assignment(self, fx, y):
        return tf.sigmoid((self.conformal.quantile_val - self.conformal.scorefn(fx, y) )/self.T)

    def loss(self, opt_state: OptimisationState):
        y_enc = opt_state.y_enc
        set_size = tf.constant(0.0)
        
        for label in self.conformal.input_properties.get_labels():
            sla = self.soft_label_assignment(y_enc, label)
            set_size = set_size + sla
            
        return tf.maximum(tf.constant(0.0), set_size - self.kappa)            
