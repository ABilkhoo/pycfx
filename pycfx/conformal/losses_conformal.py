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


class SetSizeLossLCP:

    def __init__(self, conformal: BaseLCP, T=0.4, kappa=1, **kwargs):
        pass

    def mixed_distance_squared(t1, t2, input_properties):
        """
        Calculate the sqaured mixed L2 distance between l1 and l2.
        Mixed distance is the L2 norm of individual feature distances, which is the difference between features for numeric and scaled ordinal features, and for categorical features it is 1 if the category matches and 0 otherwise.
        If InputProperties is None, all features are treated as numeric. 
        """
        pass

    def gaussian_kernel(t1, t2, kernel_bandwidth, input_properties):
        """
        Given two tensors (a calibration point and the test point) compute the weight of the calibration point according to a Gaussian kernel with specified bandwidth
        """
        pass

    def recalibrate_smooth(self, test_point_tensor):
        """
        Recalibrate for a new test_point_tensor: updates self.quantile_val using the LCP procedure, a smooth version of what is by the self.conformal BaseLCP object.
        """
        pass

    def soft_label_assignment(self, fx, y):
        """
        Obtain the soft label assignment sigmoid(quantile_val(x) - scorefn(fx, y) / T)
        """
        pass

    def loss(self, opt_state: OptimisationState):
        """
        Obtain the LCP smooth conformal set size for opt_state.x_enc: max(0, sum(soft_label_assignment) - kappa)
        """
        pass

@register_loss(SetSizeLossLCP, BACKEND_PYTORCH)
class SetSizeLossLCP_PyTorch(SetSizeLossLCP):
    """PyTorch implementation of SetSizeLossLCP"""
    def mixed_distance_squared(t1, t2, input_properties):
        if input_properties is None:
            return torch.sum((t1 - t2) ** 2, dim=1)
        else:
            t1 = torch.atleast_2d(t1)
            t2 = torch.atleast_2d(t2)

            num_diffs = t1[:, input_properties.non_cat_idx] - t2[:, input_properties.non_cat_idx]
            dists_sq = torch.sum(num_diffs ** 2, dim=1)

            for group in input_properties.categorical_groups:
                cat_group_dists = torch.sum(0.5 * (t1[:, group] - t2[:, group]) ** 2, dim=1)
                dists_sq = dists_sq + cat_group_dists

            return dists_sq

    def gaussian_kernel(t1, t2, kernel_bandwidth, input_properties):
        diff = 0

        if input_properties is None:
            diff = torch.dist(t1, t2, 2)
        else:
            dist_elements = torch.zeros(input_properties.n_distinct_features, requires_grad=True)
            j = 0

            for i in range(input_properties.n_features):
                if input_properties.feature_classes[i] != 'categorical':
                    dist_elements[j] += t1[i] - t2[i]
                    j += 1

            for group in input_properties.categorical_groups:
                group_vals = t1[group]
                group_vals_2 = t2[group]
                dist_elements[j] = torch.sum(group_vals == group_vals_2) 
                j += 1

        return torch.exp(-1 * diff / (2 * kernel_bandwidth * kernel_bandwidth))


    def __init__(self, conformal: BaseLCP, T=0.4, kappa=1, **kwargs):
        self.conformal = conformal
        self.T = T
        self.kappa = torch.tensor(kappa)

        assert self.conformal.X_calib_encoded is not None
        self.X_calib_enc = torch.tensor(self.conformal.X_calib_encoded, dtype=torch.float32, requires_grad=True)

        self.calib_len = len(self.conformal.y_calib)
        self.weights = torch.zeros((self.calib_len + 1,), dtype=torch.float32, requires_grad=True)
        self.kernel_input_properties = self.conformal.input_properties if self.conformal.dim_reduction is None else None
        
        self.sorted_indices = torch.tensor(np.argsort(self.conformal.scores))
        self.scores = torch.tensor(self.conformal.scores, dtype=torch.float32, requires_grad=True)[self.sorted_indices]
        self.scores = torch.cat((self.scores, torch.tensor([float('inf')], requires_grad=True, dtype=torch.float32)))


    def recalibrate_smooth(self, test_point_tensor):
        dists_sq = SetSizeLossLCP.mixed_distance_squared(self.X_calib_enc, test_point_tensor, self.kernel_input_properties)
        weights = torch.exp(-dists_sq / (2 * self.conformal.kernel_bandwidth ** 2))
        weights /= 1 + torch.sum(weights)

        weights = weights[self.sorted_indices]

        cumulative_prob = torch.cumsum(weights, dim=0)
        quantile_index = torch.searchsorted(cumulative_prob, 1.0 - self.conformal.alpha)
        self.quantile_val = self.scores[quantile_index]

        return self.quantile_val

    def soft_label_assignment(self, fx, y):
        return torch.sigmoid((self.quantile_val - self.conformal.scorefn(fx, y) )/self.T)

    def loss(self, opt_state: OptimisationState):
        x_enc = opt_state.x_enc
        y_enc = opt_state.y_enc

        device = x_enc.device
        self.X_calib_enc = self.X_calib_enc.to(device)
        self.weights = self.weights.to(device)
        self.sorted_indices = self.sorted_indices.to(device)
        self.scores = self.scores.to(device)

        self.quantile_val = self.recalibrate_smooth(x_enc)

        set_size = torch.tensor(0.0, requires_grad=True, device=device)
        
        for label in self.conformal.input_properties.get_labels():
            sla = self.soft_label_assignment(y_enc, label)
            set_size = set_size + sla

            
        return torch.max(torch.tensor(0.0, device=device), set_size - self.kappa)            
        