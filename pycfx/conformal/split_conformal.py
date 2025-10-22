"""
pycfx/conformal/split_conformal.py
SplitConformalPrediction
"""

from pycfx.datasets.input_properties import InputProperties
from pycfx.models.abstract_model import AbstractModel
from pycfx.models import DecisionTreeSKLearn, RandomForestSKLearn
from pycfx.datasets.dim_reduction import DimensionalityReduction
from pycfx.conformal.score_fns import get_scorefn, MILPEncodableScoreFn

import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from typing import Type


class SplitConformalPrediction:
    """
    Wrapper over AbstractModel to compute vanilla (split) conformal prediction intervals
    """

    def __init__(self, model: AbstractModel, input_properties: InputProperties, config: dict, save_path: Path=None, use_pretrained: bool=True):
        """
        Initialise with model, dataset input_properties and config.
        Alter the following with the config dict: {'alpha':0.05, 'scorefn_name':'linear2', 'dim_reduction':None}
        Set save_path and use_pretrained to save and re-use calibration predictions.
        
        Use a custom score function by adding it to the SCOREFN_REGISTRY (see pycfx/conformal/score_fns.py)
        """
        self.model = model
        self.input_properties = input_properties

        self.config = config
        self.alpha = self.config.get('alpha', 0.05)
        self.scorefn_name = self.config.get('scorefn_name', 'linear2')
        self.scorefn = get_scorefn(self.scorefn_name)()
        self.dim_reduction: DimensionalityReduction = self.config.get('dim_reduction', None)

        self.is_calibrated = False
        self.scores = None
        self.calib_preds = None

        self.save_path = save_path
        self.use_pretrained = use_pretrained

    def name(self, exclude=None) -> str:
        """
        Get the name of this class, including specified config
        """
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Type):
                    return str(obj)
                if isinstance(obj, DimensionalityReduction):
                    return obj.name()
                return super().default(obj)
        
        config = self.config

        if exclude:
            config = {k: v for k, v in config.items() if k != exclude}
    
        return self.__class__.__name__ + json.dumps(config, separators=(',', ':'), cls=CustomEncoder)

    def get_scores(self, X_calib: np.ndarray, y_calib: np.ndarray) -> np.array:
        """
        Get nonconformity scores for calibation points (X_calib, y_calib) using the configured scorefn. If model has a save_dir, and use_pretrained=True, these scores will be saved. 
        """
        if self.model.save_dir:
            scores_path = self.model.save_dir / f"scores_{self.scorefn_name}.npz"

            if scores_path.is_file() and self.use_pretrained:
                loaded = np.load(scores_path)
                if len(loaded["X_calib"]) == len(X_calib) and np.all(loaded["X_calib"] == X_calib) and np.all(loaded["y_calib"] == y_calib):
                    self.calib_preds = loaded["calib_preds"]
                    self.scores = loaded["scores"]
                    return self.scores

        preds = self.model.predict(X_calib)
        
        scores = np.zeros((len(y_calib),))
        for j in range(len(y_calib)):
            scores[j] = self.scorefn(preds[j], y_calib[j]) 

        if self.save_path:
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(scores_path, X_calib=X_calib, y_calib=y_calib, calib_preds=preds, scores=scores)

        self.calib_preds = preds
        self.scores = scores
        return self.scores

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray, test_point: np.ndarray=None) -> np.float32:
        """
        Calibrate the SplitConformalPrediction instance using calibration data (X_calib, y_calib), returns the quantile value. test_point is ignored.
        """
        scores = self.get_scores(X_calib, y_calib)
        self.quantile_val = np.quantile(scores, 1 - self.alpha)
        self.is_calibrated = True
        return self.quantile_val

    def predict(self, X: np.ndarray) -> list:
        """
        Obtain a conformal prediction interval for a single example X
        """
        y_labels = self.input_properties.get_labels()
        prediction = self.model.predict(X.reshape(1, -1))[0]
        pred_interval = []
        for element in y_labels:
            score = self.scorefn(prediction, element)
            if score <= self.quantile_val:
                pred_interval.append(element)

        return pred_interval
    
    def predict_batch(self, X: np.ndarray) -> list[list]:
        """
        Obtain conformal prediction intervals for a batch of examples X
        """
        y_labels = self.input_properties.get_labels()
        predictions = self.model.predict(X)
        pred_intervals = []

        for i in range(len(predictions)):
            pred_interval = []
            for element in y_labels:
                score = self.scorefn(predictions[i], element)
                if score <= self.quantile_val:
                    pred_interval.append(element)
            pred_intervals.append(pred_interval)

        return pred_intervals

    def gp_set_conformal_prediction_constraint(self, grb_model: gp.Model, output_vars: gp.MVar, input_vars: gp.MVar) -> gp.MVar:
        """
        Give the prediction of the model (in the output_vars MVar), add an MVar constrained to the nonconformity score of each class.
        input_vars is unused for SplitConformalPrediction.
        Returns the scores MVar
        """
        if not isinstance(self.scorefn, MILPEncodableScoreFn):
            raise ValueError("Score function not MILP encodable")

        #Conformal prediction constraint:
        # For target class:
            # score of found cf <= quantile
        # For other classes:
            # score of found cf >= quantle

        num_classes = self.input_properties.n_targets

        self.scores_c = grb_model.addVars(num_classes, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="scores") 
        self.scorefn.gp_encode_scores(grb_model, self.scores_c, output_vars, self.input_properties)

        return self.scores_c


    #Check move to non-numeric labels
    def gp_set_singleton_constraint(self, grb_model: gp.Model, target_class: int) -> None:
        """
        Constrain the conformal set size to be a singleton containing the target class.
        
        Note: Call gp_set_conformal_prediction_constraint first to set the scores for each class.
        TODO: remove dependency on one-hot labels here.  

        Returns the singleton set size constraints. The target class can be changed by removing these constraints from the model, then calling this function again.
        """
        singleton_constraints = []

        for i in range(self.input_properties.n_targets):
            if i == target_class:
                c = grb_model.addConstr(self.scores_c[i] <= self.quantile_val, name=f"target_{i}")
                singleton_constraints.append(c)
            else:
                c = grb_model.addConstr(self.scores_c[i] >= self.quantile_val + 1e-6, name=f"other_{i}")
                singleton_constraints.append(c)

        return singleton_constraints