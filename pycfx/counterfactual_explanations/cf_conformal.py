"""
pycfx/counterfactual_explanations/cf_conformal.py
Compute a CFX for the model using the CONFEX approach.
"""

from pycfx.counterfactual_explanations.cf_generator import CounterfactualGenerator
import gurobipy as gp
from gurobipy import GRB
from pycfx.datasets.input_properties import InputProperties
from pycfx.conformal.split_conformal import SplitConformalPrediction
from pycfx.conformal.conformal_helpers import median_pairwise_distances
import numpy as np
from pycfx.models.abstract_model import MILPEncodableModel
import os

class ConformalCF(CounterfactualGenerator):
    """
    Compute a CFX for the model using the CONFEX approach: find the minimal distance point which has a conformal prediction set consisting of the target class only.
    """
    def __init__(self, model: MILPEncodableModel, input_properties: InputProperties, config: dict, save_dir=None, use_pregenerated=True):
        """
        Initialise with `model`, dataset `input_properties` and config dict.
        Set save_dir and use_pregenerated to allow generators to save and reuse data.
        Specify within the config `conformal_class`, a subtype of SplitConformalPrediction, and `conformal_config`, the config dict to pass to the `conformal_class` when initialising it.
        """
        super().__init__(model, input_properties, config, save_dir, use_pregenerated)

        #assert conformal.is_calibrated, "Conformal prediction must be calibrated before generating counterfactuals."
        self.conformal_class = self.config.get('conformal_class', SplitConformalPrediction)
        self.conformal_config = self.config.get('conformal_config', {'alpha': 0.05})

    def setup(self, X_train: np.ndarray, y_train: np.ndarray, X_calib: np.ndarray, y_calib: np.ndarray):
        self.conformal = self.conformal_class(self.model, self.input_properties, config=self.conformal_config, save_path=self.save_dir, use_pretrained=self.use_pregenerated)

        if self.conformal.dim_reduction:
            self.conformal.dim_reduction.setup(self.model, self.input_properties, X_train, y_train, self.save_dir, self.use_pregenerated)

        self.conformal.calibrate(X_calib, y_calib)
        
        self.grb_model = gp.Model("model")
        # self.grb_model.setParam('TimeLimit', 60)
        self.grb_model.setParam('OutputFlag', 1)

        self.input_vars, self.input_mvar = self.input_properties.gp_set_input_var_constraints(self.grb_model)
        self.output_vars = self.model.gp_set_model_constraints(self.grb_model, self.input_mvar)

        self.conformal.gp_set_conformal_prediction_constraint(self.grb_model, self.output_vars, self.input_mvar)

        self.singleton_constraints = []
        self.distance_constrs = []

        self.distance_vars = self.grb_model.addVars(len(self.input_vars), lb=0, name="d")
        self.grb_model.setObjective(gp.quicksum(self.distance_vars), GRB.MINIMIZE)

            
    def _set_instance(self, x: np.ndarray, y_target: int):
        """
        Remove constraints dependent on the input factual and target class, and replace them with new `x` and `y_target`. 
        Prevents rebuiling the whole model for a new instance, and can use a warm start to the optimisation.
        """

        self.grb_model.remove(self.singleton_constraints)
        
        self.singleton_constraints = self.conformal.gp_set_singleton_constraint(self.grb_model, y_target)

        self.grb_model.remove(self.distance_constrs)
        self.distance_constrs = []
        for i in range(len(self.input_vars)):
            c1 = self.grb_model.addConstr(-1 * self.distance_vars[i] <= self.input_vars[i] - x[i], name=f"abs_pos_{i}")
            c2 = self.grb_model.addConstr(self.input_vars[i] - x[i] <= self.distance_vars[i], name=f"abs_neg_{i}")
            self.distance_constrs.extend([c1, c2])
        

    def generate_counterfactual(self, x: np.ndarray, y_target: int) -> np.array:
        self._set_instance(x, y_target)
        self.grb_model.optimize()

        if self.grb_model.status == GRB.OPTIMAL:
            # for var in self.grb_model.getVars():
            #     print(var.VarName, '=', var.X)

            return self.check_solution(self.input_mvar, y_target)
        else:
            return np.full_like(self.input_mvar, np.nan)
        