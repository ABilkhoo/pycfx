"""
pycfx/counterfactual_explanations/cf_mindist.py
Compute the minimal distance CFX using MILP.
"""

from pycfx.counterfactual_explanations.cf_generator import CounterfactualGenerator
from pycfx.datasets.input_properties import InputProperties
from pycfx.models.abstract_model import MILPEncodableModel

import numpy as np
from gurobipy import GRB
import gurobipy as gp
from pathlib import Path

class MinDistanceCF(CounterfactualGenerator):
    """
    MinDistanceCF: Compute the minimal distance CFX using MILP.
    """
    def __init__(self, model: MILPEncodableModel, input_properties: InputProperties, config: dict, save_dir: Path=".", use_pregenerated: bool=True):
        """
        Initialise with `model`, dataset `input_properties` and config dict.
        Set save_dir and use_pregenerated to allow generators to save and reuse data.
        Alter config "db_distance" for to alter threshold for logits to be assignment to the target class.
        """
        super().__init__(model, input_properties, config, save_dir, use_pregenerated)
        self.db_distance = self.config.get("db_distance", 0.05)

        self.grb_model = gp.Model("model")
        # self.grb_model.setParam('OutputFlag', 0)
        self.grb_model.setParam('TimeLimit', 60)

        self.input_vars, self.input_mvar = self.input_properties.gp_set_input_var_constraints(self.grb_model)
        self.output_vars = self.model.gp_set_model_constraints(self.grb_model, self.input_mvar)
        self.clf_constraints = []

        self.distance_vars = self.grb_model.addVars(len(self.input_vars), lb=0, name="d")
        self.distance_constrs = []

        self.grb_model.setObjective(gp.quicksum(self.distance_vars), GRB.MINIMIZE)
       
    def _set_instance(self, x: np.ndarray, y_target: int):
        """
        Remove constraints dependent on the input factual and target class, and replace them with new `x` and `y_target`. 
        Prevents rebuiling the whole model for a new instance, and can use a warm start to the optimisation.
        """
        self.grb_model.remove(self.clf_constraints)
        self.clf_constraints = self.model.gp_set_classification_constraint(self.grb_model, self.output_vars, target_class=y_target, db_distance=self.db_distance)

        self.grb_model.remove(self.distance_constrs)
        self.distance_constrs = []
        for i in range(len(self.input_vars)):
            c1 = self.grb_model.addConstr(-1 * self.distance_vars[i] <= self.input_vars[i] - x[i], name=f"abs_pos_{i}")
            c2 = self.grb_model.addConstr(self.input_vars[i] - x[i] <= self.distance_vars[i], name=f"abs_neg_{i}")
            self.distance_constrs.extend([c1, c2])

    def generate_counterfactual(self, x: np.ndarray, y_target: int):
        self._set_instance(x, y_target)
        self.grb_model.optimize()

        if self.grb_model.status == GRB.OPTIMAL:
            return self.check_solution(self.input_mvar, y_target)
        else:
            return np.full_like(self.input_mvar, np.nan)