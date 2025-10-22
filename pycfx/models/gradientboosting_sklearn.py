"""
pycfx/models/gradientboosting_sklearn.py
Gradient boosting classifier
"""

from pycfx.models.abstract_model import MILPEncodableModel
from pycfx.datasets.input_properties import InputProperties

import gurobipy as gp
from gurobipy import GRB
from gurobi_ml.sklearn import add_decision_tree_regressor_constr
from gurobi_ml import add_predictor_constr
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostingSKLearn(MILPEncodableModel):
    """
    Wrapper over a SKLearn GradientBoostingClassifier.
    """

    def __init__(self, config: dict, input_properties: InputProperties, random_state: int=0):
        """
        Initialise with dataset `input_properties` and `config`
        The config key `random_state` is set to a fixed 0, and other SKLearn arguments to GradientBoostingClassifier (e.g. n_estimators) can be stated in the config.
        """
         
        super().__init__(config, input_properties, random_state)
        self.config = {
            "n_estimators": config.get('n_estimators', 5),
            "learning_rate": config.get('learning_rate', 0.1),
            "max_depth": config.get('max_depth', 3),
        }

        self.random_state = random_state
        self.model = GradientBoostingClassifier(random_state=random_state, **self.config)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, x: np.ndarray, softmax: bool=False) -> np.array:
        return_single = False
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            return_single = True

        if softmax:
            pred = self.model.predict_proba(x)[0]
            return pred[0] if return_single else pred

        n_classes = self.model.n_classes_
        learning_rate = self.model.learning_rate

        if n_classes == 2:
            score_class_1 = 0

            raw_scores = np.zeros((x.shape[0], 2)) 

            for stage in self.model.estimators_:
                tree_output = stage[0].predict(x) 
                raw_scores[:, 1] += learning_rate * tree_output

            return raw_scores[0] if return_single else raw_scores
        else:
            raw_scores = np.zeros((x.shape[0], n_classes))
            for stage in self.model.estimators_: 
                for k in range(n_classes):
                    raw_scores[:, k] += learning_rate * stage[k].predict(x)
                
            return raw_scores[0] if return_single else raw_scores
    
    def gp_set_model_constraints(self, grb_model: gp.Model, input_mvar: gp.MVar) -> gp.MVar:

        n_classes = self.model.n_classes_
        learning_rate = self.model.learning_rate

        if n_classes == 2:
            raw_score = grb_model.addVar(lb=-GRB.INFINITY, name="raw_score")
            tree_outputs = []

            for stage_idx, est in enumerate(self.model.estimators_): 
                tree_out = grb_model.addVar(lb=-GRB.INFINITY, name=f"tree_out_stage{stage_idx}")
                add_predictor_constr(grb_model, est[0], input_mvar, tree_out, epsilon=1e-5)
                tree_outputs.append(learning_rate * tree_out)

            grb_model.addConstr(raw_score == sum(tree_outputs), name="logit_binary")

            raw_scores = grb_model.addMVar(2, lb=-GRB.INFINITY, name="raw_scores")
            # Set to [0, logit] since that will softmax to correct output
            grb_model.addConstr(raw_scores[0] == 0)
            grb_model.addConstr(raw_scores[1] == raw_score)

            return raw_scores
        else:
            raw_scores = grb_model.addMVar(n_classes, lb=-GRB.INFINITY, name="raw_scores")

            tree_outputs_by_class = [[] for _ in range(n_classes)]

            for stage_idx, stage in enumerate(self.model.estimators_):
                for k in range(n_classes):
                    tree_out = grb_model.addVar(lb=-GRB.INFINITY, name=f"tree_out_stage{stage_idx}_class{k}")

                    add_decision_tree_regressor_constr(grb_model, stage[k], input_mvar, tree_out)

                    tree_outputs_by_class[k].append(learning_rate * tree_out)
                    
            for k in range(n_classes):
                grb_model.addConstr(raw_scores[k] == sum(tree_outputs_by_class[k]), name=f"logit_class_{k}")

            return raw_scores

    def gp_set_classification_constraint(self, grb_model: gp.Model, output_vars: gp.MVar, target_class: int, db_distance=1e-6) -> None:
        clf_constraints = []
        for i in range(output_vars.shape[0]):
            if i != target_class:
                clf_constraints.append(grb_model.addConstr(output_vars[target_class] >= output_vars[i] + 1e-6))
            
        return clf_constraints


