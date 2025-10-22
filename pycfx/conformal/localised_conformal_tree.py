"""
pycfx/conformal/localised_conformal_tree.py
CONFEXTree implementation
"""


from pycfx.conformal.split_conformal import SplitConformalPrediction
from pycfx.library.tree_model_encoding import _leaf_formulation
from pycfx.conformal.conformal_helpers import median_pairwise_distances
from pycfx.models.abstract_model import AbstractModel
from pycfx.datasets.input_properties import InputProperties

import numpy as np
from itertools import product
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from typing import Dict, Tuple

class CONFEXTreeNode:
    """
    CONFEXTreeNode: Tree used by CONFEXTree
    """

    def __init__(self, parent, left_node, right_node, value, threshold, feature, points, scores, alpha, max_distance, points_mask=None, inf_quantile=True, global_quantile=None):
        """
        Initialize a CONFEXTreeNode with its parent, children (left_node, right_node), feature currently split by (feature, threshold), 
        all points (points), scores for each point (scores), target quantile (alpha), splitting criterion (max_distance), 
        binary mask of points assigned to current leaf (points_mask), whether to include inf the quantile (inf_quantile), whether to consider a global quantile 
        """
        self.parent = parent
        self.left_node = left_node
        self.right_node = right_node
        self.value = value
        self.threshold = threshold
        self.feature = feature
        self.points = points
        self.scores = scores
        self.alpha = alpha
        self.max_distance = max_distance
        self.is_leaf = False
        self.inf_quantile = inf_quantile
        self.global_quantile = global_quantile

        if points_mask is None:
            self.points_mask = np.ones((len(self.points),), dtype=bool)
        else:
            self.points_mask = points_mask

        self.n_dims = self.points.shape[1]

    def get_node_points(self):
        """
        Retrieve the points that belong to this node based on the mask.
        """
        return self.points[self.points_mask]

    def leaf_criterion_fulfilled(self):
        """
        Check if the node satisfies the leaf criterion based on the maximum width of the points.
        """
        points = self.get_node_points()
        widths = np.ptp(points, axis=0)

        return np.max(widths) < self.max_distance 
        
    def splitting_threshold(self):
        """
        Determine the feature and threshold for splitting the node.
        """
        points = self.get_node_points()
        widths = np.ptp(points, axis=0)
        split_feature = np.argmax(widths)

        split_feature_values = self.get_node_points()[:, split_feature]
        threshold = (np.min(split_feature_values) + np.max(split_feature_values)) / 2

        return split_feature, threshold

    def generate(self):
        """
        Recursively generate the tree by splitting nodes until the leaf criterion is fulfilled.
        """
        if not self.leaf_criterion_fulfilled():
            self.feature, self.threshold = self.splitting_threshold()
            points_mask_left = np.logical_and(self.points_mask, self.points[:, self.feature] <= self.threshold)
            points_mask_right = np.logical_and(self.points_mask, self.points[:, self.feature] > self.threshold)
            
            self.left_node = CONFEXTreeNode(self, None, None, None, None, None, self.points, self.scores, self.alpha, self.max_distance, points_mask_left, self.inf_quantile, self.global_quantile)
            self.right_node = CONFEXTreeNode(self, None, None, None, None, None, self.points, self.scores, self.alpha, self.max_distance, points_mask_right, self.inf_quantile, self.global_quantile)
            self.left_node.generate()
            self.right_node.generate()
        else:
            self.is_leaf = True
            points = self.get_node_points()
            points_max = np.max(points, axis=0)
            points_min = np.min(points, axis=0)
            self.centre = (points_max + points_min) / 2

            if len(self.scores[self.points_mask]) <= 1:
                self.value = 100
            else:
                if self.inf_quantile:
                    self.value = np.quantile(np.append(self.scores[self.points_mask], 100), 1-self.alpha, method="closest_observation")
                else:
                    self.value = np.quantile(self.scores[self.points_mask], 1-self.alpha)

            if self.global_quantile is not None:
                self.value = max(self.value, self.global_quantile)
    
    def predict(self, X):
        """
        Predict the value for a given input or batch of inputs.
        """
        if isinstance(X[0], (list, np.ndarray)):
            return np.array([self._predict_single(x) for x in X])
        else:
            return self._predict_single(X)

    def _predict_single(self, x, v=False):
        """
        Predict the value for a single input by traversing the tree.
        """
        if self.is_leaf:
            if v:
                print(f"centre, {self.centre}, quantile, {self.value}")
                print(np.linalg.norm(self.centre - x, ord=np.inf))

            if np.linalg.norm(self.centre - x, ord=np.inf) > 0.5 * self.max_distance:
                if v:
                    print("inf")
                    return self
                return np.inf
            
            if v:
                print(self.value)
                return self
            
            return self.value
        elif x[self.feature] <= self.threshold:
            return self.left_node._predict_single(x, v=v)
        else:
            return self.right_node._predict_single(x, v=v)

    def size_tree(self):
        """
        Calculate the total number of nodes and leaves in the tree.
        """
        n_nodes = 0
        n_leaves = 0
        
        def recurse(node):
            nonlocal n_nodes, n_leaves
            n_nodes += 1

            if node.value:
                print(len(node.get_node_points()))
                n_leaves += 1
            
            if not node.value:
                recurse(node.left_node)
                recurse(node.right_node)

        recurse(self)
        return n_nodes, n_leaves
        
    def generate_sklearn_representation(self):
        """
        Generate a scikit-learn compatible representation of the tree. Passed to gurobi-machinelearning's tree encoding.
        """
        children_left = []
        children_right = []
        feature = []
        threshold = []
        value = []

        def recurse(node):
            node_id = len(children_left)

            if node.is_leaf:  # Leaf node
                children_left.append(-1)
                children_right.append(-1)
                feature.append(-2)
                threshold.append(-2.0)
                value.append(np.hstack((node.centre, node.value)).reshape(1, -1))
                return node_id

            # Decision node
            feature.append(node.feature)
            threshold.append(node.threshold)
            children_left.append(None)   
            children_right.append(None)  
            value.append(np.zeros((self.n_dims + 1,)).reshape(1, -1))  

            current_id = node_id
            left_id = recurse(node.left_node)
            right_id = recurse(node.right_node)
            children_left[current_id] = left_id
            children_right[current_id] = right_id

            return current_id

        recurse(self)

        dict = {
            'children_left': np.array(children_left, dtype=np.int64),
            'children_right': np.array(children_right, dtype=np.int64),
            'feature': np.array(feature, dtype=np.int64),
            'threshold': np.array(threshold, dtype=np.float32),
            'value': np.array(value, dtype=np.float32),
            'capacity': len(children_left),
            'n_features': self.n_dims
        }

        return dict

    def visualise(self, indent=""):
        """
        Visualise the tree structure in a human-readable format.
        """
        if self.is_leaf:
            print(indent + f"Leaf(value={self.value})")
        else:
            print(indent + f"Node(feature={self.feature}, threshold={self.threshold})")
            print(indent + "├── True (≤)")
            self.left_node.visualise(indent + "│   ")
            print(indent + "└── False (>)")
            self.right_node.visualise(indent + "    ")


class ConformalCONFEXTree(SplitConformalPrediction):
    """CONFEXTree: Compute a tree-based """

    def __init__(self, model: AbstractModel, input_properties: InputProperties, config: dict, save_path: Path=None, use_pretrained: bool=False):
        """
        Initialize the ConformalCONFEXTree with the model, input properties, and configuration.
        Alter the following with the config dict: {'kernel_bandwidth':1, 'scorefn_name':'linear2', 'dim_reduction':None,  'kernel_bandwidth_scaling': True, cat_groups_to_ignore:[]}
        
        Setting dim_reduction will the CONFEXTree over the lower dimensional space given by dim_reduction.encode
        Setting cat_groups_to_ignore will not split on the specific indices corresponding a one-hot categorical feature.
        If kernel_bandwidth_scaling is True, the specified kernel bandwidth is used as a multiplier with the median pairwise distance of the calibration points, else it is used as-is.
        """
        super().__init__(model, input_properties, config, save_path, use_pretrained)

        self.kernel_bandwidth = self.config.get('kernel_bandwidth', 1)
        self.kernel_bandwidth_scaling = self.config.get('kernel_bandwidth_scaling', True)
        self.inf_quantile = self.config.get('inf_quantile', True)
        self.global_quantile = self.config.get('global_quantile', False)
        self.cat_groups_to_ignore = self.config.get('idx_cat_groups_to_ignore', [])
        self.min_quantile = None
        self.is_calibrated = False
        self.med_pairwise_distance = None

        self.gs = [group for (idx, group) in enumerate(self.input_properties.categorical_groups) if idx not in self.cat_groups_to_ignore]
        self.categorical_values_combinations = product(*self.gs)
        
        self.trees = {}

        if self.dim_reduction:
            self.numeric_mask = np.arange(self.dim_reduction.target_dim)
        else:
            self.numeric_mask = self.input_properties.non_cat_idx
        

    def get_tree(self) -> Dict[Tuple, CONFEXTreeNode]:
        """
        Retrieve the generated CONFEX trees after calibration.
        """
        assert self.is_calibrated
        return self.trees

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray, test_point: np.ndarray=None) -> np.float32:
        """
        Calibrate the CONFEX tree using calibration data and scores.
        """
        scores = self.get_scores(X_calib, y_calib)
        if self.global_quantile:
            self.global_quantile = np.quantile(scores, 1-self.alpha)

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

        if self.dim_reduction:
            Xs = self.dim_reduction.encode(X_calib)
            tree = CONFEXTreeNode(None, None, None, None, None, None, Xs, scores, self.alpha, self.kernel_bandwidth, inf_quantile=self.inf_quantile, global_quantile=self.global_quantile)
            tree.generate()
            self.trees[()] = tree
            self.is_calibrated = True
            return

        for val in self.categorical_values_combinations:
            included_indices = np.any(X_calib, axis=1)

            if len(val) != 0:
                included_indices = np.any(X_calib[:, np.atleast_1d(val)] == 1, axis=1)

            X_calib_included = X_calib[included_indices]
            scores_included = scores[included_indices]
            X_calib_numeric = X_calib_included[:, self.numeric_mask]
            
            tree = CONFEXTreeNode(None, None, None, None, None, None, X_calib_numeric, scores_included, self.alpha, self.kernel_bandwidth, inf_quantile=self.inf_quantile, global_quantile=self.global_quantile)
            tree.generate()
            self.trees[val] = tree

        self.is_calibrated = True
    
    def predict(self, X: np.ndarray) -> list:
        """
        Predict the conformal prediction interval for a single input.
        """
        assert self.is_calibrated

        y_labels = self.input_properties.get_labels()
        prediction = self.model.predict(X.reshape(1, -1))[0]
        pred_interval = []

        if self.dim_reduction:
            tree = self.trees[()]
            quantile_val = tree.predict(self.dim_reduction.encode(X).reshape(1, -1))[0]
        else:
            cat_group_combiantion = tuple([i[np.argmax(X[i])] for i in self.gs])
            tree = self.trees[cat_group_combiantion]
            quantile_val = tree.predict(X[self.numeric_mask].reshape(1, -1))[0]

        for element in y_labels:
            score = self.scorefn(prediction, element)
            if score <= quantile_val:
                pred_interval.append(element)

        return pred_interval
        
    def predict_batch(self, X: np.ndarray) -> list[list]:
        """
        Return the conformal prediction intervals for a batch of inputs.
        """
        assert self.is_calibrated

        y_labels = self.input_properties.get_labels()
        predictions = self.model.predict(X)
        quantiles = np.zeros((X.shape[0],))

        if self.dim_reduction:
            X_enc = self.dim_reduction.encode(X)
            tree = self.trees[()]
            quantiles = tree.predict(X_enc)
        else:
            for i in range(X.shape[0]):
                cat_group_combiantion = tuple([j[np.argmax(X[i][j])] for j in self.gs])
                tree = self.trees[cat_group_combiantion]
                quantiles[i] = tree.predict(X[i][self.numeric_mask].reshape(1, -1))[0]
            
        pred_intervals = []

        for i in range(len(predictions)):
            pred_interval = []
            for element in y_labels:
                score = self.scorefn(predictions[i], element)
                if score <= quantiles[i]:
                    pred_interval.append(element)
            pred_intervals.append(pred_interval)

        return pred_intervals
    
    def gp_set_conformal_prediction_constraint(self, grb_model: gp.Model, output_vars: gp.MVar, input_vars: gp.MVar):
        """
        Add conformal prediction constraints to a Gurobi optimisation model.
        """
        if self.scorefn_name not in ['linear', 'linear2', 'linear_normalised'] :
            raise ValueError("Can only use linear scorefn in MILP")

        #Conformal prediction constraint:
        # For target class:
            # score of found cf <= quantile
        # For other classes:
            # score of found cf >= quantile

        assert self.is_calibrated

        if self.dim_reduction:
            tree = self.trees[()]
            output_mvar = grb_model.addMVar(shape=(1 + tree.n_dims,), lb=-float('inf'), name="tree_output_dimr")
            tree_dict = tree.generate_sklearn_representation()
            input_vars_reduced = self.dim_reduction.gp_dim_encoding(grb_model, input_vars)
            _leaf_formulation(grb_model, input_vars_reduced, output_mvar, tree_dict, epsilon=0.01)
        else:
            tree_combinations, trees = list(self.trees.keys()), list(self.trees.values())
            tree_indicators = grb_model.addMVar((len(self.trees),), vtype=GRB.BINARY)

            input_vars_reduced = input_vars[self.input_properties.non_cat_idx]
            output_mvar = grb_model.addMVar(shape=(1 + trees[0].n_dims,), lb=-float('inf'), name="tree_output_total")
            tree_outputs = []

            for i in range(len(tree_combinations)):
                tree_i = trees[i]
                tree_dict_i = tree_i.generate_sklearn_representation()
                tree_output_i = grb_model.addMVar(shape=(1 + tree_i.n_dims,), lb=-float('inf'), name=f"tree_output_{i}")
                _leaf_formulation(grb_model, input_vars_reduced, tree_output_i, tree_dict_i, epsilon=0.01)
                tree_outputs.append(tree_output_i)

                combination = tree_combinations[i]

                for j in combination:
                    grb_model.addConstr(tree_indicators[i] <= input_vars[j])
                
                grb_model.addConstr(tree_indicators[i] >= gp.quicksum(input_vars[j] for j in combination) - len(combination) + 1)
            
            grb_model.addConstr(gp.quicksum(tree_indicators) == 1)
            grb_model.addConstr(output_mvar == gp.quicksum(tree_indicators[i] * tree_outputs[i] for i in range(len(tree_combinations)))) 
        

        quantile_val = output_mvar[-1]
        centre = output_mvar[:-1]

        self.quantile_val = grb_model.addVar(lb=-float('inf'), name="quantile_val")
        grb_model.addConstr(quantile_val == self.quantile_val)

        differences = grb_model.addMVar(shape=centre.shape, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="differences")
        distance = grb_model.addVar(name="distance")

        if self.dim_reduction:
            for i in range(differences.shape[0]):
                grb_model.addConstr(differences[i] == centre[i] - input_vars_reduced[i])
        else:
            for i in range(differences.shape[0]):
                grb_model.addConstr(differences[i] == centre[i] - input_vars[i])
        
        grb_model.addConstr(distance == gp.norm(differences, np.inf), f"gp_get_weights__distances_{i}")
        grb_model.addConstr(distance <= 0.5 * self.kernel_bandwidth)

        num_classes = self.input_properties.n_targets

        self.scores_c = grb_model.addVars(num_classes, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="scores") 

        self.scorefn.gp_encode_scores(grb_model, self.scores_c, output_vars, self.input_properties)
