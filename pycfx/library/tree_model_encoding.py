"""
pycfx/library/tree_model_encoding.py
Leaf formulation of SKLearn Decision Trees from the gurobi-machinelearning, licenced under the Apache Licence 2.0, Copyright 2021 Gurobi Optimization, LLC
Note: this code has been modified from handling decision tree regressors to decision tree classifiers.
See the original at https://github.com/Gurobi/gurobi-machinelearning/blob/main/pycfx/gurobi_ml/modeling/decision_tree/decision_tree_model.py
"""

from gurobipy import GRB
import gurobipy as gp
import numpy as np
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

def _compute_leafs_bounds(tree, epsilon):
    """Compute the bounds that define each leaf of the tree"""
    capacity = tree["capacity"]
    n_features = tree["n_features"]
    children_left = tree["children_left"]
    children_right = tree["children_right"]
    feature = tree["feature"]
    threshold = tree["threshold"]

    node_lb = -np.ones((n_features, capacity)) * GRB.INFINITY
    node_ub = np.ones((n_features, capacity)) * GRB.INFINITY

    stack = [
        0,
    ]

    while len(stack) > 0:
        node = stack.pop()
        left = children_left[node]
        if left < 0:
            continue
        right = children_right[node]
        assert left not in stack
        assert right not in stack
        node_ub[:, right] = node_ub[:, node]
        node_lb[:, right] = node_lb[:, node]
        node_ub[:, left] = node_ub[:, node]
        node_lb[:, left] = node_lb[:, node]

        node_ub[feature[node], left] = threshold[node]
        node_lb[feature[node], right] = threshold[node] + epsilon
        stack.append(right)
        stack.append(left)
    return (node_lb, node_ub)


def _leaf_formulation(
    gp_model, _input, output, tree, epsilon=0
):
    """Formulate decision tree using 'leaf' formulation

    We have one variable per leaf of the tree and a series of indicator to
    define when that leaf is reached.
    """
    _input = _input.reshape(1, -1)
    output_internal = gp_model.addMVar(output.reshape(1, -1).shape, lb=-GRB.INFINITY)
    gp_model.update()
    
    nex = _input.shape[0]
    n_features = tree["n_features"]

    # Collect leaf nodes
    leafs = tree["children_left"] < 0
    leafs_vars = gp_model.addMVar(
        (nex, sum(leafs)), vtype=GRB.BINARY, name="leafs"
    )

    (node_lb, node_ub) = _compute_leafs_bounds(tree, epsilon)
    input_ub = _input.getAttr(GRB.Attr.UB)
    input_lb = _input.getAttr(GRB.Attr.LB)

    for i, node in enumerate(leafs.nonzero()[0]):
        reachable = (input_ub >= node_lb[:, node]).all(axis=1) & (
            input_lb <= node_ub[:, node]
        ).all(axis=1)
        # Non reachable nodes
        leafs_vars[~reachable, i].setAttr(GRB.Attr.UB, 0.0)
        # Leaf node:
        rhs = output_internal[reachable, :].tolist()
        lhs = leafs_vars[reachable, i].tolist()
        values = tree["threshold"].reshape(-1, 1)[node, :]
        n_indicators = sum(reachable)
        for l_var, r_vars in zip(lhs, rhs):
            for r_var, value in zip(r_vars, values):
                gp_model.addGenConstrIndicator(l_var, 1, r_var, GRB.EQUAL, value)

        for feature in range(n_features):
            feat_lb = node_lb[feature, node]
            feat_ub = node_ub[feature, node]

            if feat_lb > -GRB.INFINITY:
                tight = (input_lb[:, feature] < feat_lb) & reachable
                lhs = leafs_vars[tight, i].tolist()
                rhs = _input[tight, feature].tolist()
                n_indicators += sum(tight)
                for l_var, r_var in zip(lhs, rhs):
                    gp_model.addGenConstrIndicator(
                        l_var, 1, r_var, GRB.GREATER_EQUAL, feat_lb
                    )

            if feat_ub < GRB.INFINITY:
                tight = (input_ub[:, feature] > feat_ub) & reachable
                lhs = leafs_vars[tight, i].tolist()
                rhs = _input[tight, feature].tolist()
                n_indicators += sum(tight)
                for l_var, r_var in zip(lhs, rhs):
                    gp_model.addGenConstrIndicator(
                        l_var, 1, r_var, GRB.LESS_EQUAL, feat_ub
                    )
        

    # We should attain 1 leaf
    gp_model.addConstr(leafs_vars.sum(axis=1) == 1)
    gp_model.addConstr(output == leafs_vars[0] @ tree['value'][leafs][:, 0])

    # gp_model.addConstr(output <= np.max(tree["value"], axis=0))
    # gp_model.addConstr(output >= np.min(tree["value"], axis=0))

    return leafs_vars
