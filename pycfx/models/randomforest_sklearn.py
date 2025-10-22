"""
pycfx/models/randomforest_sklearn.py
Random forest classifier
"""

from pycfx.models.abstract_model import AbstractModel
from pycfx.models.decisiontree_sklearn import DecisionTreeSKLearn
from pycfx.datasets.input_properties import InputProperties

from sklearn.ensemble import RandomForestClassifier
import gurobipy as gp
import numpy as np

class RandomForestSKLearn(AbstractModel):
    """
    Wrapper over SKLearn's RandomForestClassifier
    """

    def __init__(self, config: dict, input_properties: InputProperties):
        """
        Initialise with dataset `input_properties` and `config`
        The config key `random_state` is set to a fixed 0, and other SKLearn arguments to RandomForestClassifier (e.g. n_estimators) can be stated in the config.
        """

        super().__init__(config, input_properties)

        self.n_estimators = self.config.get("n_estimators", 5)
        self.max_depth = self.config.get("max_depth", 30)
        self.max_n_leaves = self.config.get("max_n_leaves", None)

        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state, max_leaf_nodes=self.max_n_leaves)

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, x: np.ndarray) -> np.array:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            return self.model.predict_proba(x)[0]
        return self.model.predict_proba(x)
    
    def gp_set_model_constraints(self, grb_model: gp.Model, input_mvar: gp.MVar) -> gp.MVar:
        tree_outputs = []
        n = self.model.n_estimators

        for i in range(n):
            tree = self.model.estimators_[i]
            tree_enc = DecisionTreeSKLearn(config={})
            tree_enc.load_external(tree)
            tree_output = tree_enc.gp_set_model_constraints(grb_model, input_mvar)
            tree_outputs.append(tree_output)

        output_mvar = grb_model.addMVar(shape=(self.model.n_classes_,), lb=0, ub=1, name="output_var")

        for j in range(self.model.n_classes_):  # Iterate over the 3 elements in the vector
            grb_model.addConstr(sum(tree_outputs[i][j] for i in range(n)) / n == output_mvar[j])

        return output_mvar
    
    def gp_set_classification_constraint(self, grb_model: gp.Model, output_vars: gp.MVar, target_class: int, db_distance=1e-6) -> None:
        clf_constraints = []
        for i in range(output_vars.shape[0]):
            if i != target_class:
                clf_constraints.append(grb_model.addConstr(output_vars[target_class] >= output_vars[i] + db_distance))

        return clf_constraints
