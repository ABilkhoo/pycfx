"""
pycfx/conformal/other/localised_conformal_others.py
Split Localised Conformal Prediction (Han et. al. 2022, see https://arxiv.org/abs/2206.13092)
"""


from pycfx.datasets.input_properties import InputProperties
from typing import Literal
import numpy as np
import pandas as pd
from pycfx.conformal.milp_utils import *
from sklearn.ensemble import RandomForestRegressor
from pycfx.conformal.localised_conformal_lcp import BaseLCP, SplitConformalPrediction
from pycfx.conformal.conformal_helpers import *
from pycfx.conformal.kernels import get_kernel

class SplitLCP(SplitConformalPrediction):

    def __init__(self, model, input_properties, config, save_path=None, use_pretrained=None):
        super().__init__(model, input_properties, config, save_path, use_pretrained)

        self.kernel_name = self.config.get('kernel_name', 'box_l1')
        self.kernel_bandwidth = self.config.get('kernel_bandwidth', 1)
        self.kernel = get_kernel(self.kernel_name)
        self.sample_threshold = self.config.get('sample_threshold', 1000)
        self.dim_reduction = self.config.get('dim_reduction', None)
        self.scores_nonlocalised = None
        self.calib_points = None

        self.train_scores_sorted = None

    def calculate_train_scores(self):
        if self.sample_threhold is not None and len(self.X_train) > self.sample_threhold:
            train_data, train_labels = sample_points(train_data, train_labels, self.sample_threhold)
        else:
            train_data = self.X_train
            train_labels = self.y_train

        preds = self.model.predict(train_data)
        calib_len = len(train_labels)
        
        scores = np.zeros((calib_len,))
        for j in range(calib_len):
            scores[j] = self.scorefn(preds[j], train_labels[j]) 

        sorted_indices = np.argsort(scores)

        self.train_scores_sorted = scores[sorted_indices]
        self.X_train_sorted = train_data[sorted_indices]
        self.y_train_sorted = train_labels[sorted_indices]

        if self.dim_reduction is not None:
            self.X_train_sorted = self.dim_reduction.encode(self.X_train_sorted)

    
    def calculate_qfhat(self, test_point):
        if self.train_scores_sorted is None:
            self.calculate_train_scores()

        scores = self.train_scores_sorted
        scores = np.append(scores, [np.inf], axis=0)
        train_data = self.X_train_sorted

        if self.dim_reduction is not None:
            test_point = self.dim_reduction.encode(test_point)

        kernel_values = np.zeros((train_data.shape[0] + 1,))

        for i in range(train_data.shape[0]):
            train_datapoint = train_data[i]
            kernel_values[i] = self.kernel(train_datapoint, test_point, self.kernel_bandwidth)
        kernel_values[-1] = 1
        
        weights = kernel_values.copy()
        if np.sum(weights) > 0:
            weights /= np.sum(weights)

        q_f_hat = float('inf')
        cumulative_prob = 0.0
        for i in range(len(weights)):
            cumulative_prob += weights[i]
            if cumulative_prob >= 1.0 - self.alpha:
                q_f_hat = scores[i]
                break

        return q_f_hat
  
    def calibrate(self, X_calib, y_calib, test_point=None):
        scores = self.get_scores(X_calib, y_calib)

        for j in range(X_calib.shape[0]):
            scores[j] -= self.calculate_qfhat(X_calib[j])

        self.quantile_val = np.quantile(scores, 1-self.alpha)
        self.is_calibrated = True

    def predict(self, X):
        assert self.is_calibrated

        y_labels = range(self.input_properties.n_targets)
        prediction = self.model.predict(X.reshape(1, -1))[0]

        q_fhat = self.calculate_qfhat(X)

        pred_interval = []
        for element in y_labels:
            score = self.scorefn(prediction, element)
            score -= q_fhat
            
            if score <= self.quantile_val:
                pred_interval.append(element)

        return pred_interval
    
    def predict_batch(self, X):
        assert self.is_calibrated

        y_labels = range(self.input_properties.n_targets)
        predictions = self.model.predict(X)
        pred_intervals = []

        for i in range(len(predictions)):
            pred_interval = []
            q_fhat = self.calculate_qfhat(X[i])
            for element in y_labels:
                score = self.scorefn(predictions[i], element)
                score -= q_fhat
                if score <= self.quantile_val:
                    pred_interval.append(element)
            pred_intervals.append(pred_interval)

        return pred_intervals

    def gp_set_conformal_prediction_constraint(self, grb_model: gp.Model, output_vars: gp.MVar, input_vars: gp.MVar, target_class: int):
        if self.kernel_name != 'box_l1' and self.scorefn_name not in ['linear', 'linear2']:
            raise ValueError("Can only use linear scorefn and box_l1 kernel in MILP")
        
        assert self.is_calibrated

        scores = self.train_scores_sorted
        # scores = np.append(scores, [np.max(scores) + 1000], axis=0) ##
        scores_mvar = gp_set_np_mvar(grb_model, scores, "scores")

        sorted_train_data = self.X_train_sorted
        values_mvar = gp_set_np_mvar(grb_model, sorted_train_data, "X_train")

        if self.dim_reduction:
            input_vars_reduced = self.dim_reduction.gp_dim_encoding(grb_model, input_vars)
            weights_c_mvar = gp_get_weights(grb_model, values_mvar, input_vars_reduced, self.kernel_bandwidth)
        else:
            weights_c_mvar = gp_get_weights(grb_model, values_mvar, input_vars, self.kernel_bandwidth)

        # quantile_val = gp_get_weighted_quantile_new(grb_model, scores_mvar, weights_c_mvar, 1-self.alpha)
        self.slcp_quantile = gp_get_weighted_quantile(grb_model, scores_mvar, weights_c_mvar, 1-self.alpha)

        num_classes = self.input_properties.n_targets

        self.scores_c = grb_model.addVars(num_classes, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="scores_test") 

        self.scorefn.gp_encode_scores(grb_model, self.scores_c, output_vars, self.input_properties)

        
    def gp_set_singleton_constraint(self, grb_model: gp.Model, target_class: int):
        singleton_constraints = []

        for i in range(self.input_properties.n_targets):
            if i == target_class:
                c = grb_model.addConstr(self.scores_c[i] - self.slcp_quantile <= self.quantile_val, name=f"target_{i}")
                singleton_constraints.append(c)
            else:
                c = grb_model.addConstr(self.scores_c[i] - self.slcp_quantile >= self.quantile_val + 1e-6, name=f"other_{i}")
                singleton_constraints.append(c)


