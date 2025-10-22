"""
pycfx/conformal/score_fns.py
Score functions used by SplitConformalPrediction, and subclasses.
"""


from pycfx.datasets.input_properties import InputProperties

from abc import ABC, abstractmethod
import numpy as np
import gurobipy as gp
from gurobipy import GRB

"""Registry of all scorefunctions, which are instances of ScoreFn"""
SCOREFN_REGISTRY = {}

class ScoreFn(ABC):
    """
    Abstract ScoreFn class
    """

    @abstractmethod
    def __call__(self, fx_logits, y_correct):
        """
        Return the score of the point x given the model's prediction of x `fx_logits` for the particular class `y_correct`
        """
        pass
     
class MILPEncodableScoreFn(ScoreFn):
    """
    Abstract score which has a corresponding MILP encoding.
    """

    @abstractmethod
    def gp_encode_scores(self, grb_model, scores, output_vars, input_properties):
        """
        Define constraints for the variable `scores` (shape: n_classes) to represent the nonconformity scores based on the model's predictions `output_vars` for each class.
        """
        pass

def register_scorefn(key: str):
    """Decorator to add a ScoreFn subclass to the SCOREFN_REGISTRY with key `key`"""

    def decorator(subclass):
        assert issubclass(subclass, ScoreFn)
        SCOREFN_REGISTRY[key] = subclass
        return subclass
    return decorator

def get_scorefn(key: str) -> ScoreFn:
    """Get the score function class with key `key` from the registry"""
    return SCOREFN_REGISTRY[key]

@register_scorefn("softmax")
class SoftmaxScoreFn(ScoreFn):
    """
    Softmax score: s(x, y) = 1 - logit of class y
    """
    def __call__(self, fx_logits: np.ndarray, y_correct: np.ndarray):
        fx_softmax = np.exp(fx_logits) / np.sum(np.exp(fx_logits))
        return 1 - fx_softmax[y_correct]
    
@register_scorefn("linear")
class LinearScoreFn(MILPEncodableScoreFn):
    """
    Linear score: s(x, y) = max logit - logit of class y 
    """
    def __call__(self, fx_logits: np.ndarray, y_correct: np.ndarray):
        return -1 * fx_logits[y_correct] + max(fx_logits)

    def gp_encode_scores(self, grb_model: gp.Model, scores: np.ndarray, output_vars: np.ndarray, input_properties: InputProperties):
        max_logit = grb_model.addVar(lb=-float('inf'), vtype=GRB.CONTINUOUS, name="max_logits")
        grb_model.addConstr(max_logit == gp.max_(*output_vars))
        grb_model.addConstrs(scores[i] == -1 * output_vars[i] + max_logit for i in range(input_properties.n_targets))

    
@register_scorefn("linear2")
class Linear2ScoreFn(MILPEncodableScoreFn):
    """
    Linear score: s(x, y) = max logit (exluding class y) - logit of class y 
    For binary classification, this is the difference between the two logits.
    """
    def __call__(self, fx_logits, y_correct):
        return -1 * fx_logits[y_correct] + max([fx_logits[i] for i in range(len(fx_logits)) if i != y_correct])
     
    def gp_encode_scores(self, grb_model: gp.Model, scores: np.ndarray, output_vars: np.ndarray, input_properties: InputProperties):
        if input_properties.n_targets == 2:
            grb_model.addConstr(scores[0] == -1 * output_vars[0] + output_vars[1])
            grb_model.addConstr(scores[1] == -1 * output_vars[1] + output_vars[0])

        else:
            max_other_logit = grb_model.addMVar(shape=(output_vars.shape[0],), lb=-float('inf'), vtype=GRB.CONTINUOUS, name="max_logits")
            for i in range(output_vars.shape[0]):
                grb_model.addConstr(max_other_logit[i] == gp.max_([output_vars[j] for j in range(output_vars.shape[0]) if j != i]))

            grb_model.addConstrs(scores[i] == -1 * output_vars[i] + max_other_logit[i] for i in range(input_properties.n_targets))

    
@register_scorefn("linear_normalised")
class LinearNormalisedScoreFn(MILPEncodableScoreFn):
    """
    Linear normalised score: s(x, y) = 1 - logit of class y
    Note that fx_logits add up to 1 and heuristically look like probabilities for each class, e.g. with a random forest model.
    """
    def __call__(self, fx_logits: np.ndarray, y_correct: np.ndarray):
        return 1 - fx_logits[y_correct]

    def gp_encode_scores(self, grb_model: gp.Model, scores: np.ndarray, output_vars: np.ndarray, input_properties: InputProperties):
        grb_model.addConstrs(scores[i] == 1 - output_vars[i] for i in range(input_properties.n_targets))
  