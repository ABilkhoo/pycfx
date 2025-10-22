"""
pycfx/conformal/other/localised_conformal_others.py
Regression-tree based LCP (alternative to LCP)
"""


from pycfx.conformal.split_conformal import SplitConformalPrediction
from pycfx.datasets.input_properties import InputProperties
from pycfx.datasets.dim_reduction import DimensionalityReduction
from pycfx.conformal.milp_utils import *
from pycfx.conformal.conformal_helpers import *
from pycfx.conformal.localised_conformal_lcp import *
from pycfx.conformal.other.localised_conformal_slcp import *

from typing import Literal
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from gurobi_ml.sklearn import add_random_forest_regressor_constr

class BaseLCP_R(BaseLCP):

    def __init__(self, model, input_properties, config, save_path=None, use_pretrained=None):
        super().__init__(model, input_properties, config, save_path, use_pretrained)
        self.qr_trained = None
    
    def calibrate(self, X_calib, y_calib, test_point=None):
        super().calibrate(X_calib, y_calib, test_point)
        self.train_r_model()

    def train_r_model(self, factor=None):
        if factor == None:
            if len(self.X_calib) > 1000:
                factor = 1
            else:
                factor = 5

        grid_points = generate_grid_points(self.input_properties, self.X_calib, self.dim_reduction, factor=factor)

        qr_x = []
        qr_y = []

        self.scores_nonlocalised[np.isinf(self.scores_nonlocalised)] = np.max(self.scores_nonlocalised[~np.isinf(self.scores_nonlocalised)])

        for t in range(grid_points.shape[0]):
            test_point = grid_points[t]
            calib_len = len(self.y_calib)
            weights = np.zeros((calib_len+1,))

            for j in range(calib_len):
                weights[j] = self.kernel(self.calib_points[j], test_point, self.kernel_bandwidth)

            weights[calib_len] = self.kernel(test_point, test_point, self.kernel_bandwidth)
            weights /= np.sum(weights)

            sorted_indices = np.argsort(self.scores_nonlocalised)
            scores = self.scores_nonlocalised[sorted_indices]
            weights = weights[sorted_indices]

            quantile_val = np.max(scores) + 1000
            cumulative_prob = 0.0
            for i in range(calib_len):
                cumulative_prob += weights[i]
                if cumulative_prob >= 1.0 - self.alpha:
                    quantile_val = scores[i]
                    break
            
            qr_x.append(test_point)
            qr_y.append(quantile_val)
        
        qr_x = np.array(qr_x)
        qr_y = np.array(qr_y)

        qr_model = RandomForestRegressor(n_estimators=4, random_state=2)
        qr_model.fit(qr_x, qr_y)
        self.qr_x = qr_x
        self.qr_y = qr_y

        self.qr_model = qr_model

        self.qr_trained = True
        return qr_model
    
    def predict(self, X):
        assert self.is_calibrated
        assert self.qr_trained

        y_labels = range(self.input_properties.n_targets)
        prediction = self.model.predict(X.reshape(1, -1))[0]

        if self.dim_reduction:
            quantile_val_pred = self.qr_model.predict(self.dim_reduction.encode(X).reshape(1, -1))[0]
        else:
            quantile_val_pred = self.qr_model.predict(X.reshape(1, -1))[0]
        pred_interval = []

        for element in y_labels:
            score = self.scorefn(prediction, element)
            if score <= quantile_val_pred:
                pred_interval.append(element)

        return pred_interval

    def predict_batch(self, X):
        assert self.is_calibrated
        assert self.qr_trained

        y_labels = range(self.input_properties.n_targets)
        predictions = self.model.predict(X)
        pred_intervals = []

        if self.dim_reduction:
            quantiles = self.qr_model.predict(self.dim_reduction.encode(X))
        else:
            quantiles = self.qr_model.predict(X)

        for i in range(len(predictions)):
            pred_interval = []
            for element in y_labels:
                score = self.scorefn(predictions[i], element)
                quantile_val = quantiles[i]
                if score <= quantile_val:
                    pred_interval.append(element)
            pred_intervals.append(pred_interval)

        return pred_intervals


    def gp_set_conformal_prediction_constraint(self, 
                                               grb_model: gp.Model, 
                                               output_vars: gp.MVar, 
                                               input_vars: gp.MVar):
        if self.kernel_name != 'box_l1' and self.scorefn_name not in ['linear', 'linear2']:
            raise ValueError("Can only use linear scorefn and box_l1 kernel in MILP")
        
        assert self.is_calibrated
        assert self.qr_trained

        self.quantile_val = grb_model.addVar(lb=float('-inf'), name="quantile_val_approx")

        if self.dim_reduction:
            input_vars_reduced = self.dim_reduction.gp_dim_encoding(grb_model, input_vars)
            add_random_forest_regressor_constr(grb_model, self.qr_model, input_vars_reduced, self.quantile_val)
        else:
            add_random_forest_regressor_constr(grb_model, self.qr_model, input_vars, self.quantile_val)

        num_classes = self.input_properties.n_targets

        self.scores_c = grb_model.addVars(num_classes, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="scores_test") 

        self.scorefn.gp_encode_scores(grb_model, self.scores_c, output_vars, self.input_properties)



class SplitLCP_R(SplitLCP):
    def __init__(self, model, 
                       alpha: float, 
                       input_properties: InputProperties, 
                       scorefn_name: Literal['softmax', 'linear'], 
                       kernel_name: Literal['box_l1', 'box_l2', 'gaussian'],
                       kernel_bandwidth: float, 
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       dim_reduction: DimensionalityReduction = None,):
        super().__init__(model, alpha, input_properties, scorefn_name, kernel_name, kernel_bandwidth, X_train, y_train, dim_reduction=dim_reduction)
        self.qr_trained = False

    
    def train_r_model(self):
        scores = self.train_scores_sorted
        scores = np.append(scores, [1000], axis=0)
        train_data = self.X_train_sorted

        if factor == None:
            if len(train_data) > 1000:
                factor = 1
            else:
                factor = 5

        grid_points = generate_grid_points(self.input_properties, train_data, self.dim_reduction, factor=factor)

        qr_x = []
        qr_y = []

        for i in range(grid_points.shape[0]):
            kernel_values = np.zeros((train_data.shape[0] + 1,))

            for j in range(train_data.shape[0]):
                train_datapoint = train_data[j]
                kernel_values[j] = self.kernel(train_datapoint, grid_points[i], self.kernel_bandwidth)
            kernel_values[-1] = 1 

            weights = kernel_values.copy()
            if np.sum(weights) > 0:
                weights /= np.sum(weights)

            q_f_hat = 1000
            cumulative_prob = 0.0
            for k in range(len(weights)):
                cumulative_prob += weights[k]
                if cumulative_prob >= 1.0 - self.alpha:
                    q_f_hat = scores[k]
                    break
            
            qr_x.append(grid_points[i])
            qr_y.append(q_f_hat)

        
        qr_x = np.array(qr_x)
        qr_y = np.array(qr_y)

        qr_model = RandomForestRegressor(n_estimators=2, random_state=2)
        qr_model.fit(qr_x, qr_y)
        self.qr_x = qr_x
        self.qr_y = qr_y

        self.qr_model = qr_model

        self.qr_trained = True
        return qr_model
    

    def calibrate(self, X_calib, y_calib, test_point=None):
        super().calibrate(X_calib, y_calib, test_point)
        self.train_r_model()


    def predict(self, X):
        assert self.is_calibrated
        assert self.qr_trained

        y_labels = range(self.input_properties.n_targets)
        prediction = self.model.predict(X.reshape(1, -1))[0]

        if self.dim_reduction:
            q_fhat_pred = self.qr_model.predict(self.dim_reduction.encode(X).reshape(1, -1))[0]
        else:
            q_fhat_pred = self.qr_model.predict(X.reshape(1, -1))[0]

        pred_interval = []

        for element in y_labels:
            score = self.scorefn(prediction, element)
            if score - q_fhat_pred <= self.quantile_val:
                pred_interval.append(element)

        return pred_interval
    
    def predict_batch(self, X):
        assert self.is_calibrated
        assert self.qr_trained

        y_labels = range(self.input_properties.n_targets)
        predictions = self.model.predict(X)
        pred_intervals = []

        if self.dim_reduction:
            qhats = self.qr_model.predict(self.dim_reduction.encode(X))
        else:
            qhats = self.qr_model.predict(X)

        for i in range(len(predictions)):
            pred_interval = []
            q_fhat = qhats[i]
            for element in y_labels:
                score = self.scorefn(predictions[i], element)
                score -= q_fhat
                if score <= self.quantile_val:
                    pred_interval.append(element)
            pred_intervals.append(pred_interval)

        return pred_intervals
    
    def calculate_qfhat(self, test_point):
        if self.train_scores_sorted is None:
            self.calculate_train_scores()

        if not self.qr_trained:
            self.train_r_model()
        
        q_f_hat = self.qr_model.predict(test_point.reshape(1, -1))[0]

        return q_f_hat

    def gp_set_conformal_prediction_constraint(self, grb_model: gp.Model, output_vars: gp.MVar, input_vars: gp.MVar, target_class: int):
        if self.kernel_name != 'box_l1' and self.scorefn_name not in ['linear', 'linear2']:
            raise ValueError("Can only use linear scorefn and box_l1 kernel in MILP")
        
        assert self.is_calibrated

        self.slcp_quantile = grb_model.addVar(lb=float('-inf'), name="quantile_val_approx")

        if self.dim_reduction:
            input_vars_reduced = self.dim_reduction.gp_dim_encoding(grb_model, input_vars)
            add_random_forest_regressor_constr(grb_model, self.qr_model, input_vars_reduced, self.slcp_quantile)
        else:
            add_random_forest_regressor_constr(grb_model, self.qr_model, input_vars, self.slcp_quantile)

        num_classes = self.input_properties.n_targets

        self.scores_c = grb_model.addVars(num_classes, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="scores_test") 

        self.scorefn.gp_encode_scores(grb_model, self.scores_c, output_vars, self.input_properties)



