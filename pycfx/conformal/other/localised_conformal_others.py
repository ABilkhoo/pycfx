"""
pycfx/conformal/other/localised_conformal_others.py
Other LCP methods: ConformalisedQuantileClassification, LoCart, LoForest
"""

from pycfx.conformal.split_conformal import SplitConformalPrediction
from pycfx.datasets.input_properties import InputProperties

import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.tree import DecisionTreeRegressor
from itertools import product
from typing import Literal
import copy
from gurobi_ml.sklearn import add_decision_tree_regressor_constr

import gurobipy as gp
from gurobipy import GRB

from pycfx.datasets.dim_reduction import DimensionalityReduction

class ConformalisedQuantileClassification(SplitConformalPrediction):
    def __init__(self, model, 
                       alpha: float, 
                       input_properties: InputProperties, 
                       scorefn_name: Literal['softmax', 'linear'], 
                       X_train: np.ndarray,
                       y_train: np.ndarray):
        super().__init__(model, alpha, input_properties, scorefn_name)
        self.X_train, self.y_train = X_train, y_train 
        self.qr_trained = False

    def train_r_model(self):
        train_data = self.X_train
        scores = np.zeros((train_data.shape[0],))

        train_preds = self.model.predict(train_data)
        for i in range(train_data.shape[0]):
            scores[i] = self.scorefn(train_preds[i], self.y_train[i])

        qrf = RandomForestQuantileRegressor(random_state=2, ccp_alpha=self.alpha)
        qrf.fit(train_data, scores)

        self.qr_model = qrf
        self.qr_trained = True

        return qrf
   
    def calibrate(self, X_calib, y_calib, test_point=None):
        if not self.qr_trained:
            self.train_r_model()

        calib_data = X_calib
        scores = np.zeros((calib_data.shape[0],))
        qr_val = self.qr_model.predict(X_calib)

        calib_preds = self.model.predict(calib_data)
        for i in range(calib_data.shape[0]):
            scores[i] = self.scorefn(calib_preds[i], y_calib[i]) - qr_val[i]

        
        self.quantile_val = np.quantile(scores, 1 - self.alpha)
        self.is_calibrated = True
        self.scores = scores
        return self.quantile_val
    
    def predict_batch(self, X):
        y_labels = range(self.input_properties.n_targets)
        predictions = self.model.predict(X)
        qr_val = self.qr_model.predict(X)
        pred_intervals = []

        for i in range(len(predictions)):
            pred_interval = []
            for element in y_labels:
                score = self.scorefn(predictions[i], element) - qr_val[i]
                if score <= self.quantile_val:
                    pred_interval.append(element)
            pred_intervals.append(pred_interval)

        return pred_intervals
    


class ConformalisedLocart(SplitConformalPrediction):
    def __init__(self, model, 
                       alpha: float, 
                       input_properties: InputProperties, 
                       scorefn_name: Literal['softmax', 'linear'], 
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       dim_reduction: DimensionalityReduction = None):
        super().__init__(model, alpha, input_properties, scorefn_name)
        self.X_train, self.y_train = X_train, y_train 
        self.qr_trained = False
        self.dim_reduction = dim_reduction


    def train_r_model(self):
        train_data = self.X_train
        scores = np.zeros((train_data.shape[0],))

        train_preds = self.model.predict(train_data)
        for i in range(train_data.shape[0]):
            scores[i] = self.scorefn(train_preds[i], self.y_train[i]) 

        self.scores = scores
        tree = DecisionTreeRegressor(min_samples_leaf=30, random_state=10)

        if self.dim_reduction:
            train_data = self.dim_reduction.encode(train_preds)

        tree.fit(train_data, scores)

        self.qr_model = tree
        
        self.quantile_tree = copy.deepcopy(tree)
        leaf_indices = tree.apply(train_data)

        unique_leaves = np.unique(leaf_indices)
        leaf_quantiles = {}

        for leaf in unique_leaves:
            leaf_scores = scores[leaf_indices == leaf]
            leaf_quantiles[leaf] = np.quantile(leaf_scores, 1 - self.alpha)

        for node in range(self.quantile_tree.tree_.node_count):
            if self.quantile_tree.tree_.children_left[node] == self.quantile_tree.tree_.children_right[node]:  # is leaf
                leaf_id = node
               
                leaf_order = np.where(
                    (self.quantile_tree.tree_.children_left == self.quantile_tree.tree_.children_right)
                )[0]
                # Map node index to unique_leaves order
                idx_in_unique = np.where(leaf_order == node)[0][0]
                leaf_val = leaf_quantiles[unique_leaves[idx_in_unique]]
                self.quantile_tree.tree_.value[node][0][0] = leaf_val
        
        self.qr_trained = True

        return self.quantile_tree
   
    def calibrate(self, X_calib, y_calib, test_point=None):
        if not self.qr_trained:
            self.train_r_model()
        self.is_calibrated = True

    
    def predict_batch(self, X):
        y_labels = range(self.input_properties.n_targets)
        predictions = self.model.predict(X)

        if self.dim_reduction:
            qr_vals = self.quantile_tree.predict(self.dim_reduction.encode(X))
        else:
            qr_vals = self.quantile_tree.predict(X)

        pred_intervals = []

        for i in range(len(predictions)):
            pred_interval = []
            for element in y_labels:
                score = self.scorefn(predictions[i], element) 
                if score <= qr_vals[i]:
                    pred_interval.append(element)
            pred_intervals.append(pred_interval)

        return pred_intervals
    

    def gp_set_conformal_prediction_constraint(self, 
                                               grb_model: gp.Model, 
                                               output_vars: gp.MVar, 
                                               input_vars: gp.MVar,
                                               target_class: int):
        
        if self.scorefn_name not in ['linear', 'linear2']:
            raise ValueError("Can only use linear scorefn in MILP")
        
        assert self.qr_trained

        quantile_val_tree = grb_model.addVar(lb=float('-inf'), name="quantile_val_tree")

        if self.dim_reduction:
            input_vars_reduced = self.dim_reduction.gp_dim_encoding(grb_model, input_vars)
            add_decision_tree_regressor_constr(grb_model, self.quantile_tree, input_vars_reduced, quantile_val_tree)
        else:
            add_decision_tree_regressor_constr(grb_model, self.quantile_tree, input_vars, quantile_val_tree)

        num_classes = self.input_properties.n_targets

        scores = grb_model.addVars(num_classes, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="scores_test") 

        self.scorefn.gp_encode_scores(grb_model, self.scores_c, output_vars, self.input_properties)

        for i in range(num_classes):
            if i == target_class:
                grb_model.addConstr(scores[i] <= quantile_val_tree - 1e-6, name=f"target_{i}")
            else:
                grb_model.addConstr(scores[i] >= quantile_val_tree + 1e-6, name=f"other_{i}")
        


class ConformalisedLoForest(SplitConformalPrediction):
    def __init__(self, model, 
                       alpha: float, 
                       input_properties: InputProperties, 
                       scorefn_name: Literal['softmax', 'linear'], 
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       dim_reduction: DimensionalityReduction = None):
        super().__init__(model, alpha, input_properties, scorefn_name)
        self.X_train, self.y_train = X_train, y_train 
        self.qr_trained = False
        self.dim_reduction = dim_reduction


    def train_r_model(self):
        train_data = self.X_train
        scores = np.zeros((train_data.shape[0],))

        train_preds = self.model.predict(train_data)
        for i in range(train_data.shape[0]):
            scores[i] = self.scorefn(train_preds[i], self.y_train[i]) 

        self.scores = scores
        tree = DecisionTreeRegressor(min_samples_leaf=30, random_state=10)

        if self.dim_reduction:
            train_data = self.dim_reduction.encode(train_preds)

        tree.fit(train_data, scores)

        self.qr_model = tree
        
        self.quantile_tree = copy.deepcopy(tree)
        leaf_indices = tree.apply(train_data)

        unique_leaves = np.unique(leaf_indices)
        leaf_quantiles = {}

        for leaf in unique_leaves:
            leaf_scores = scores[leaf_indices == leaf]
            leaf_quantiles[leaf] = np.quantile(leaf_scores, 1 - self.alpha)

        for node in range(self.quantile_tree.tree_.node_count):
            if self.quantile_tree.tree_.children_left[node] == self.quantile_tree.tree_.children_right[node]:  # is leaf
                leaf_id = node
               
                leaf_order = np.where(
                    (self.quantile_tree.tree_.children_left == self.quantile_tree.tree_.children_right)
                )[0]
                # Map node index to unique_leaves order
                idx_in_unique = np.where(leaf_order == node)[0][0]
                leaf_val = leaf_quantiles[unique_leaves[idx_in_unique]]
                self.quantile_tree.tree_.value[node][0][0] = leaf_val
        
        self.qr_trained = True

        return self.quantile_tree
   
    def calibrate(self, X_calib, y_calib, test_point=None):
        if not self.qr_trained:
            self.train_r_model()
        self.is_calibrated = True

    
    def predict_batch(self, X):
        y_labels = range(self.input_properties.n_targets)
        predictions = self.model.predict(X)

        if self.dim_reduction:
            qr_vals = self.quantile_tree.predict(self.dim_reduction.encode(X))
        else:
            qr_vals = self.quantile_tree.predict(X)

        pred_intervals = []

        for i in range(len(predictions)):
            pred_interval = []
            for element in y_labels:
                score = self.scorefn(predictions[i], element) 
                if score <= qr_vals[i]:
                    pred_interval.append(element)
            pred_intervals.append(pred_interval)

        return pred_intervals
    

    def gp_set_conformal_prediction_constraint(self, 
                                               grb_model: gp.Model, 
                                               output_vars: gp.MVar, 
                                               input_vars: gp.MVar,
                                               target_class: int):
        
        if self.scorefn_name not in ['linear', 'linear2']:
            raise ValueError("Can only use linear scorefn in MILP")
        
        assert self.qr_trained

        quantile_val_tree = grb_model.addVar(lb=float('-inf'), name="quantile_val_tree")

        if self.dim_reduction:
            input_vars_reduced = self.dim_reduction.gp_dim_encoding(grb_model, input_vars)
            add_decision_tree_regressor_constr(grb_model, self.quantile_tree, input_vars_reduced, quantile_val_tree)
        else:
            add_decision_tree_regressor_constr(grb_model, self.quantile_tree, input_vars, quantile_val_tree)

        num_classes = self.input_properties.n_targets

        scores = grb_model.addVars(num_classes, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="scores_test") 

        self.scorefn.gp_encode_scores(grb_model, self.scores_c, output_vars, self.input_properties)

        for i in range(num_classes):
            if i == target_class:
                grb_model.addConstr(scores[i] <= quantile_val_tree - 1e-6, name=f"target_{i}")
            else:
                grb_model.addConstr(scores[i] >= quantile_val_tree + 1e-6, name=f"other_{i}")
        



