"""
pycfx/models/decisiontree_sklearn.py
Decision Tree classifier
"""


from pycfx.models.abstract_model import MILPEncodableModel
from pycfx.datasets.input_properties import InputProperties
from pycfx.library.tree_model_encoding import _leaf_formulation

from sklearn.tree import DecisionTreeClassifier
import gurobipy as gp
import numpy as np

class DecisionTreeSKLearn(MILPEncodableModel):
    """
    Wrapper for SKLearn's DecisionTreeClassifier
    """

    def __init__(self, config: dict, input_properties: InputProperties=None):
        """
        Initialise with dataset `input_properties` and `config`
        The config key `random_state` is set to a fixed 0, and other SKLearn arguments to DecisionTreeClassifier (e.g. max_depth) can be stated in the config.
        """

        super().__init__(config, input_properties)
        self.config = {
            "max_depth": config.get('max_depth', None),
        }

        self.random_state = self.config.get("random_state")
        self.model = DecisionTreeClassifier(random_state=self.random_state, **self.config)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, x: np.ndarray) -> np.array:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            return self.model.predict_proba(x)[0]
        return self.model.predict_proba(x)
    
    def gp_set_model_constraints(self, grb_model: gp.Model, input_mvar: gp.MVar) -> gp.MVar:
        output_mvar = grb_model.addMVar(shape=(self.model.n_classes_,), lb=0, ub=1, name="output_var")

        tree = self.model.tree_

        tree_dict = {
            "children_left": tree.children_left,
            "children_right": tree.children_right,
            "feature": tree.feature,
            "threshold": tree.threshold,
            "value": tree.value,
            "capacity": tree.capacity,
            "n_features": tree.n_features,
        }

        _leaf_formulation(grb_model, input_mvar, output_mvar, tree_dict, epsilon=10e-5)

        return output_mvar
    
    def gp_set_classification_constraint(self, grb_model: gp.Model, output_vars: gp.MVar, target_class: int, db_distance=1e-6) -> None:
        clf_constraints = []
        for i in range(output_vars.shape[0]):
            if i != target_class:
                clf_constraints.append(grb_model.addConstr(output_vars[target_class] >= output_vars[i] + db_distance))
            
        return clf_constraints