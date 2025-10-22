"""
pycfx/conformal/localised_conformal_baselcp.py
Implementation of Localised Conformal Prediction (Guan, Leying. "Conformal prediction with localization." (2019)) with added MILP implemention
"""

from pycfx.conformal.split_conformal import SplitConformalPrediction
from pycfx.conformal.score_fns import MILPEncodableScoreFn
from pycfx.datasets.input_properties import InputProperties
from pycfx.datasets.dim_reduction import DimensionalityReduction
from pycfx.conformal.milp_utils import gp_set_np_mvar, gp_get_weighted_quantile_new, gp_get_weights
from pycfx.conformal.kernels import get_kernel
from pycfx.conformal.conformal_helpers import median_pairwise_distances, sample_points
from pycfx.models.abstract_model import AbstractModel

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path


class BaseLCP(SplitConformalPrediction):
    """
    Implementation of Localised Conformal Prediction (Guan, Leying. "Conformal prediction with localization." (2019)) with added MILP implemention
    """
    
    def __init__(self, model: AbstractModel, input_properties: InputProperties, config: dict, save_path: Path=None, use_pretrained: bool=None):
        super().__init__(model, input_properties, config, save_path, use_pretrained)

        """
        Initialise with model, dataset input_properties and config.
        Alter the following with the config dict: {'kernel_name':'box_l1', 'kernel_bandwidth': 1, 'scorefn_name':'linear2', 'dim_reduction':None, 'sample_threshold': 1000, 'kernel_bandwidth_scaling': True}
        Setting sample_threshold will use a sample of calibration points as opposed to the whole calibration set if exceeded, to make the MILP solve quicker.
        Setting dim_reduction will compute weights for the weighted quantile in the lower dimensional space given by dim_reduction.encode
        If kernel_bandwidth_scaling is True, the specified kernel bandwidth is used as a multiplier with the median pairwise distance of the calibration points, else it is used as-is.
        
        Set save_path and use_pretrained to save and re-use calibration predictions.

        Use a custom kernel by adding it to the KERNEL_REGISTRY (see pycfx/conformal/kernels.py), use a custom score function by adding it to the SCOREFN_REGISTRY (see pycfx/conformal/score_fns.py)
        """

        self.kernel_name = self.config.get('kernel_name', 'box_l1')
        self.kernel_bandwidth = self.config.get('kernel_bandwidth', 1)
        self.kernel_bandwidth_scaling = self.config.get('kernel_bandwidth_scaling', True)
        self.kernel = get_kernel(self.kernel_name)
        self.sample_threshold = self.config.get('sample_threshold', 1000)
        self.dim_reduction: DimensionalityReduction = self.config.get('dim_reduction', None)
        self.scores_nonlocalised = None
        self.med_pairwise_distance = None
        self.is_calibrated = False
        
        self.X_calib = None
        self.y_calib = None
        self.X_calib_encoded = None

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray, test_point: np.ndarray=None) -> np.float32:
        """
        Calibrate the conformal predictor around the `test_point` by taking the weighted quantile according to the specified kernel_name, kernel_bandwidth, kernel_bandwidth_scaling and optional dim_reduction.
        """

        if self.kernel_bandwidth_scaling and not self.is_calibrated:
            X_calib_r = X_calib

            if len(X_calib) > 10000:
                np.random.seed(2)
                random_indices = np.random.choice(len(X_calib), size=10000, replace=False)
                X_calib_r = X_calib[random_indices]

            if self.med_pairwise_distance is None and self.save_path and self.use_pretrained:
                dim_reduction_name = None
                if self.dim_reduction:
                    dim_reduction_name = self.dim_reduction.name()

                med_pairwise_distances_path = self.save_path / f"med_pairwise_distances_{dim_reduction_name}.npy"

                if med_pairwise_distances_path.is_file():
                    self.med_pairwise_distance = np.load(med_pairwise_distances_path)
                else:
                    self.med_pairwise_distance = median_pairwise_distances(X_calib_r, self.dim_reduction)
                    np.save(med_pairwise_distances_path, self.med_pairwise_distance)
            
            if self.med_pairwise_distance is None:
                self.med_pairwise_distance = median_pairwise_distances(X_calib_r, self.dim_reduction)
            
            self.kernel_bandwidth = self.kernel_bandwidth * self.med_pairwise_distance


        if self.sample_threshold is not None and len(X_calib) > self.sample_threshold:
            self.X_calib, self.y_calib = sample_points(X_calib, y_calib, self.sample_threshold)
        else:
            self.X_calib, self.y_calib = X_calib, y_calib

        self.scores_nonlocalised = self.get_scores(self.X_calib, self.y_calib)

        scores = np.append(self.scores, float('inf'))
        
        self.X_calib_encoded = self.X_calib
        
        kernel_input_properties = self.input_properties

        if self.dim_reduction is not None:
            self.X_calib_encoded = self.dim_reduction.encode(self.X_calib)
            test_point = self.dim_reduction.encode([test_point])[0]
            kernel_input_properties = None

        if test_point is None:
            self.is_calibrated = True
            self.quantile_val = np.quantile(scores, 1-self.alpha)
            return self.quantile_val
        
        calib_len = len(self.y_calib)
        weights = np.zeros((calib_len+1,))

        for j in range(calib_len):
            weights[j] = self.kernel(self.X_calib_encoded[j], test_point, self.kernel_bandwidth, kernel_input_properties)

        weights[calib_len] = self.kernel(test_point, test_point, self.kernel_bandwidth, kernel_input_properties)
        weights /= np.sum(weights)

        sorted_indices = np.argsort(self.scores_nonlocalised)
        scores = self.scores_nonlocalised[sorted_indices]
        weights = weights[sorted_indices]
        
        self.quantile_val = float('inf')
        cumulative_prob = 0.0
        for i in range(calib_len):
            cumulative_prob += weights[i]
            if cumulative_prob >= 1.0 - self.alpha:
                self.quantile_val = scores[i]
                break
        
        self.is_calibrated = True
        return self.quantile_val
    
    def predict_batch(self, X: np.ndarray) -> list[list]:
        """
        Obtain conformal prediction intervals for a batch of examples X. Recalibrates for each example X.
        """
        assert self.is_calibrated

        y_labels = self.input_properties.get_labels()
        predictions = self.model.predict(X)
        pred_intervals = []

        for i in range(len(predictions)):
            self.calibrate(X_calib=self.X_calib, y_calib=self.y_calib, test_point=X[i])
            pred_interval = []
            for element in y_labels:
                score = self.scorefn(predictions[i], element)
                if score <= self.quantile_val:
                    pred_interval.append(element)
            pred_intervals.append(pred_interval)

        return pred_intervals


    def gp_set_conformal_prediction_constraint(self, 
                                               grb_model: gp.Model, 
                                               output_vars: gp.MVar, 
                                               input_vars: gp.MVar):
        
        """
        Give the prediction of the model (in the output_vars MVar) for a test point (input_var), add an MVar constrained to the nonconformity score of each class.
        Also sets self.quantile_val to be constrained to the weighted quantile to allow constraining the set size to be a singleton.
        """

        if self.kernel_name not in ['box_l1', 'box_linf'] and not isinstance(self.scorefn, MILPEncodableScoreFn):
            raise ValueError("Can only use MILP encodable scorefn and box kernel in MILP")

        norm = 1        
        if self.kernel_name == "box_linf":
            norm = np.inf

        assert self.is_calibrated

        scores = self.scores_nonlocalised
        scores = scores[:-1]
        sorted_indices = np.argsort(scores)
        scores = self.scores_nonlocalised[sorted_indices]
        scores = np.append(scores, [np.max(scores) + 100], axis=0)
        scores_mvar = gp_set_np_mvar(grb_model, scores, "scores")

        sorted_Xcalib = self.X_calib_encoded[sorted_indices]
        values_mvar = gp_set_np_mvar(grb_model, sorted_Xcalib, "X_calib")
    
        if self.dim_reduction:
            input_vars_reduced = self.dim_reduction.gp_dim_encoding(grb_model, input_vars)
            weights_c_mvar = gp_get_weights(grb_model, values_mvar, input_vars_reduced, self.kernel_bandwidth, norm=norm)
        else:
            weights_c_mvar = gp_get_weights(grb_model, values_mvar, input_vars, self.kernel_bandwidth, input_properties=self.input_properties, norm=norm)
            

        self.quantile_val = gp_get_weighted_quantile_new(grb_model, scores_mvar, weights_c_mvar, 1-self.alpha)
        # quantile_val = gp_get_weighted_quantile(grb_model, scores_mvar, weights_c_mvar, 1-self.alpha)

        # Test conformal prediction
        num_classes = self.input_properties.n_targets

        self.scores_c = grb_model.addVars(num_classes, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="scores_test") 

        self.scorefn.gp_encode_scores(grb_model, self.scores_c, output_vars, self.input_properties)

