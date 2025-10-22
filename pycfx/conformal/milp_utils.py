"""
pycfx/conformal/milp_utils.py
Utilities for constructing MILP instances.
"""


import numpy as np
import gurobipy as gp
from gurobipy import GRB


def gp_set_np_mvar(grb_model: gp.Model, numbers: np.ndarray, name: str) -> gp.MVar:
    """
    Add an np.array of elements `numbers` to the grb_model as a constant MVar with name `name`.
    Returns the mvar.
    """
    mvar = grb_model.addMVar(shape=numbers.shape, lb=-float('inf'), name=name)
    grb_model.addConstr(mvar == numbers, f"gp_set_np_mvar__{name}")
    return mvar

def gp_get_weights(grb_model: gp.Model, values_mvar: gp.MVar, point_mvar: gp.MVar, threshold_val: float, bigM=10e2, input_properties = None, norm=1):
    """
    Given a set of calibration points `values_mvar` and a test point `point_mvar`, compute the weights of each calibration point in the LCP procedure using a 
    box kernel with kernel bandwidth `threshold_val` and a mixed L-`norm` feature distance. If input_properties is None, all features are considered numeric.
    Returns the weights MVar, of shape (values_mvar.shape[0])
    """
    differences_mvar = grb_model.addMVar(shape=values_mvar.shape, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="gp_get_weights__differences_mvar")

    if input_properties is None:
        differences_mvar = grb_model.addMVar(shape=values_mvar.shape, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="gp_get_weights__differences_mvar")

        for i in range(values_mvar.shape[0]):
            for j in range(values_mvar.shape[1]):
                grb_model.addConstr(differences_mvar[i][j] == values_mvar[i][j] - point_mvar[j], name=f"gp_get_weights__differences_{i},{j}")

    else:
        differences_mvar = grb_model.addMVar(shape=(values_mvar.shape[0], input_properties.n_distinct_features), lb=-float('inf'), vtype=GRB.CONTINUOUS, name="gp_get_weights__differences_mvar")

        for i in range(values_mvar.shape[0]):
            f_i = 0

            for j in range(values_mvar.shape[1]):
                if input_properties.feature_classes[j] != 'categorical':
                    grb_model.addConstr(differences_mvar[i][f_i] == values_mvar[i][j] - point_mvar[j], name=f"gp_get_weights__differences_{i},{j}")
                    f_i += 1

            for group in input_properties.categorical_groups:
                group_vals = values_mvar[i][group]
                group_vals_point = point_mvar[group]

                group_differences = grb_model.addMVar((len(group),), lb=-float('inf'))

                for k in range(len(group)):
                    group_differences[k] >= group_vals[k] - group_vals_point[k]
                    group_differences[k] >= group_vals_point[k] - group_vals[k] 

                grb_model.addConstr(2 * differences_mvar[i][f_i] == gp.quicksum(group_differences))
                f_i += 1

    distances_mvar = grb_model.addMVar(shape=values_mvar.shape[0], lb=-float('inf'), vtype=GRB.CONTINUOUS, name=f"gp_get_weights__distances_mvar")
    for i in range(values_mvar.shape[0]):
        grb_model.addConstr(distances_mvar[i] == gp.norm(differences_mvar[i], norm), f"gp_get_weights__distances_{i}")

    weights_mvar = grb_model.addMVar(shape=values_mvar.shape[0], vtype=GRB.BINARY, name="f_gp_get_weights__weights_mvar")
    for i in range(weights_mvar.shape[0]):
        grb_model.addConstr(threshold_val >= distances_mvar[i] - bigM*(1-weights_mvar[i]), name="f_gp_get_weights__def_1")
        grb_model.addConstr(threshold_val <= distances_mvar[i] + bigM*(weights_mvar[i]), name="f_gp_get_weights__def_2")
    
    return weights_mvar

def gp_get_quantile(grb_model: gp.Model, nums_mvar: gp.MVar, alpha: float, bigM = 10e2) -> gp.Var:
    """
    Given a list of numbers in nums_mvar, and a target `alpha`-quantile, returns a Var quantile_val which is constrained to the `alpha`-quantile of nums_mvar.
    """
    index = int(nums_mvar.shape[0] * alpha)

    count_below_mvar = grb_model.addMVar(shape=nums_mvar.shape, vtype=GRB.BINARY) 
    quantile_val = grb_model.addVar(lb=-float('inf'), name='quantile_val')

    is_quantile_val = grb_model.addMVar(shape=nums_mvar.shape, vtype=GRB.BINARY)
    grb_model.addConstr(gp.quicksum(is_quantile_val) == 1)
    grb_model.addConstr(gp.quicksum(is_quantile_val[i] * nums_mvar[i] for i in range(nums_mvar.shape[0])) == quantile_val)

    for i in range(nums_mvar.shape[0]):
        grb_model.addConstr(nums_mvar[i] <= quantile_val + bigM*(1-count_below_mvar[i])) 
        grb_model.addConstr(nums_mvar[i] >= quantile_val - bigM*(count_below_mvar[i]))
        
    grb_model.addConstr(gp.quicksum(count_below_mvar) == index + 1)

    return quantile_val

def gp_get_weighted_quantile(grb_model: gp.Model, scores_sorted_mvar: gp.MVar, weights_corresponding_mvar: gp.MVar, alpha: float, eps=1e-5) -> gp.Var:
    """
    Given a sorted list of numbers `scores_sorted_mvar` with their corresponding weights for computing the quantile `weights_corresponding_mvar`,
    returns a Var which is constrained to the weighted `alpha`-quantile of the numbers (in our case, nonconformity scores). Includes an additional weight for the delta at +inf.
    """

    quantile_val = grb_model.addVar(lb=-float('inf'), name="gp_get_weighted_quantile__quantile_val")

    indices_included_mvar = grb_model.addMVar(shape=scores_sorted_mvar.shape, vtype=GRB.BINARY, name="gp_get_weighted_quantile__indices_included_mvar")
    cutoff_index = grb_model.addVar(vtype=GRB.INTEGER, name="gp_get_weighted_quantile__cutoff_index")
    weights_sum = grb_model.addVar(name="gp_get_weighted_quantile__weights_sum")
    weights_total = grb_model.addVar(name="gp_get_weighted_quantile__weights_total")
    weights_sum_excl = grb_model.addVar(name="gp_get_weighted_quantile__weights_total_excl")

    grb_model.addConstr(weights_sum == gp.quicksum(weights_corresponding_mvar[i] * indices_included_mvar[i] for i in range(scores_sorted_mvar.shape[0])), name="gp_get_weighted_quantile__weights_sum")
    grb_model.addConstr(weights_total == 1 + gp.quicksum(weights_corresponding_mvar), name="gp_get_weighted_quantile__weights_total")
    grb_model.addConstr(weights_sum_excl == weights_total - weights_sum)

    bigM = scores_sorted_mvar.shape[0] 
    for i in range(indices_included_mvar.shape[0]):
        grb_model.addConstr(i - cutoff_index <= bigM * (1 - indices_included_mvar[i]), name=f"gp_get_weighted_quantile__le_cutoff_{i}")
        grb_model.addConstr(cutoff_index - i + 1 <= bigM * indices_included_mvar[i], name=f"gp_get_weighted_quantile__gt_cutoff_{i}")

    is_quantile_val = grb_model.addMVar(shape=scores_sorted_mvar.shape, vtype=GRB.BINARY, name="gp_get_weighted_quantile__is_quantile_val")
    grb_model.addConstr(cutoff_index == gp.quicksum(i * is_quantile_val[i] for i in range(scores_sorted_mvar.shape[0])), name="gp_get_weighted_quantile__quantile_index_link")
    grb_model.addConstr(1 == gp.quicksum(is_quantile_val), name="gp_get_weighted_quantile__quantile_index_link")

    grb_model.addConstr(quantile_val == gp.quicksum(is_quantile_val[i] * scores_sorted_mvar[i] for i in range(scores_sorted_mvar.shape[0])), name="gp_get_weighted_quantile__val_is_quantile_val")

    weights_sum_lb = grb_model.addVar(name="gp_get_weighted_quantile__weights_sum_lb", vtype=GRB.INTEGER)
    weights_sum_excl_lb = grb_model.addVar(name="gp_get_weighted_quantile__weights_sum_excl_lb", vtype=GRB.INTEGER)

    grb_model.addConstr(weights_sum_lb >= alpha * weights_total)
    grb_model.addConstr(weights_sum_lb + 1 <= alpha * weights_total + eps)
    grb_model.addConstr(weights_sum >= weights_sum_lb, name="gp_get_weighted_quantile__quantile_threshold")

    grb_model.addConstr(weights_sum_lb <= (1-alpha) * weights_total)
    grb_model.addConstr(weights_sum_lb + 1 >= (1-alpha) * weights_total + eps)
    grb_model.addConstr(weights_sum_excl >= weights_sum_excl_lb, name="gp_get_weighted_quantile__quantile_threshold_below")

    return quantile_val


def gp_get_weighted_quantile_new(grb_model: gp.Model, scores_sorted: gp.MVar, weights_corresponding_mvar: gp.MVar, alpha: float, eps = 1e-5, bigM = 100) -> gp.Var:
    """
    Alternate implementation of gp_get_weighted_quantile. 
    """
    quantile_val = grb_model.addVar(lb=-float('inf'), name="gp_get_weighted_quantile__quantile_val")

    cumulative_weights = grb_model.addMVar(shape=(scores_sorted.shape[0],), name="gp_get_weighted_quantile__cumulative_weights")
    
    grb_model.addConstr(cumulative_weights[0] == weights_corresponding_mvar[0])
    for i in range(1, scores_sorted.shape[0] - 1):
        grb_model.addConstr(cumulative_weights[i] == cumulative_weights[i-1] + weights_corresponding_mvar[i])
    grb_model.addConstr(cumulative_weights[-1] == cumulative_weights[-2] + 1)

    is_quantile_val = grb_model.addMVar(shape=scores_sorted.shape, vtype=GRB.BINARY, name="gp_get_weighted_quantile__is_quantile_val")
    grb_model.addConstr(1 == gp.quicksum(is_quantile_val), name="gp_get_weighted_quantile__quantile_index_link")

    weights_total = grb_model.addVar(name="gp_get_weighted_quantile__weights_total")
    grb_model.addConstr(weights_total == 1 + gp.quicksum(weights_corresponding_mvar), name="gp_get_weighted_quantile__weights_total")

    grb_model.addConstr(cumulative_weights[0] >= alpha * weights_total  + eps - bigM * (1 - is_quantile_val[0]))
    for i in range(1, cumulative_weights.shape[0]):
        grb_model.addConstr(cumulative_weights[i] >= alpha * weights_total + eps - bigM * (1 - is_quantile_val[i]))
        grb_model.addConstr(cumulative_weights[i-1] <= alpha * weights_total + bigM * (1 - is_quantile_val[i]))

    grb_model.addConstr(quantile_val == gp.quicksum([is_quantile_val[i] * scores_sorted[i] for i in range(scores_sorted.shape[0])]), name="gp_get_weighted_quantile__val_is_quantile_val")

    return quantile_val
