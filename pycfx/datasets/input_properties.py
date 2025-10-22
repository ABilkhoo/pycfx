"""
pycfx/datasets/input_properties.py
InputProperties: Information about a dataset
"""

from typing import List, Literal, Optional, Tuple, Union
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch

class InputProperties:
    """
    Information about a dataset
    """
    
    def __init__(self, 
                 feature_names: List[str], 
                 feature_classes: List[Literal['categorical', 'ordinal', 'numeric', 'ordinal_normalised']], 
                 bounds: List[Union[Optional[Tuple[float, float]], List[float]]],
                 n_targets: int,
                 categorical_groups: Union[List[List[int]], Literal['auto']] = 'auto',
                 y_onehot: bool=True
                ):
        
        """
        Initialise with a list of feature names, each of which has a corresponding feature class (categorical, ordinal, numeric, ordinal_normalised).
        Specify bounds for each: None for categorical, (min, max) for numeric, or a set of valid values for ordinal features.
        Specify the number of labels in `n_targets`, and whether these are onehot in `y_onehot`. 
        Specify indices of categorical features in categorical_groups. If not specified, these will be inferred from the feature names. 
        In this case, eature names with a common suffix followed by a _ will be treated as part of the same categorical group
        """
        self.n_targets = n_targets

        assert len(feature_names) == len(feature_classes), "Elements in feature_names must correspond to elements in feature_classes"
        self.feature_names = feature_names
        self.feature_classes = feature_classes
        self.n_features = len(feature_names)

        assert len(feature_names) == len(bounds), "Elements in feature_names must correspond to elements in bounds. Set elements to None for no bound or one-hot categorical"
        for i, bound in enumerate(bounds):
            if feature_classes[i] == 'ordinal':
                assert bound != None, "Ordinal features must have a lower and upper bound"
                assert (isinstance(bound[0], int) and isinstance(bound[1], int)) or (bound[0].is_integer() and bound[1].is_integer()), "Ordinal features must have an integer LB and UB"
            elif feature_classes[i] == 'ordinal_normalised':
                assert isinstance(bound, list) and all(0 <= b <= 1 for b in bound), "Ordinal normalised features must have bounds as a list of numbers between 0 and 1"
            if bound != None:
                assert bound[0] <= bound[1], "LB must be less than UB for each element"

        self.bound = bounds

        if categorical_groups == "auto":
            categorical_indices = [i for i in range(len(feature_classes)) if feature_classes[i] == 'categorical']
            categorical_sets = defaultdict(list)
            for index in categorical_indices:
                category_name = feature_names[index].split('_')[0]
                categorical_sets[category_name].append(index)
            self.categorical_groups = list(categorical_sets.values())
        else:
            for group in categorical_groups:
                for index in group:
                    assert index > 0 and index < len(feature_names), f"Index {index} of group {group} is out of bounds"
                    assert feature_classes[index] == 'categorical', f"Index {index} of group {group} is not a categorical feature"
        
            self.categorical_groups = categorical_groups  

        self.y_onehot = y_onehot
        self.n_distinct_features = self.n_features - sum([len(g) for g in self.categorical_groups]) + len(self.categorical_groups)
        self.all_idx = np.arange(self.n_features)

        if len(self.categorical_groups) == 0:
            self.all_cat_idx = np.array([])
            self.non_cat_idx = self.all_idx
        else:
            self.all_cat_idx = np.concatenate(self.categorical_groups)
            self.non_cat_idx = self.all_idx[~np.isin(self.all_idx, self.all_cat_idx)]

    def get_feature_details(self) -> List[List]:
        """
        Get a list of the feature details: name, class and bounds for each feature
        """
        feature_details = []
        for i in range(self.n_features):
            feature_details.append([self.feature_names[i], self.feature_classes[i], self.bound[i]])
        return feature_details
    
    def get_labels(self) -> List:
        """
        Get a list of dataset labels
        """
        return list(range(self.n_targets))
    
    def check_valid_instance(self, x: np.ndarray) -> None:
        """
        Check if `x` is a valid (properly encoded) instance.
        """

        for i in range(self.n_features):
            feature_class = self.feature_classes[i]
            bound = self.bound[i]
            feature_val = x[i]

            if feature_class == 'numeric' and bound is not None:
                assert feature_val >= bound[0] and feature_val <= bound[1], "Numeric feature {i} out of bounds"
            elif feature_class == 'ordinal' or feature_class == 'ordinal_normalised':
                assert np.any(np.isclose(feature_val, bound)), "Ordinal feature not of allowed values"
        
        for group in self.categorical_groups:
            group_vals = x[group]
            assert np.sum(group_vals == 0) == len(group) - 1, "Incorrectly encoded categorical variable"
            assert np.sum(group_vals == 1) == 1, "Incorrectly encoded categorical variable"

    def fix_encoding(self, x: np.ndarray) -> np.array:
        """
        Fix the encoding of `x` to the closest valid instance.
        """

        for i in range(self.n_features):
            feature_class = self.feature_classes[i]
            bound = self.bound[i]

            if feature_class == 'numeric' and bound is not None:
                if isinstance(x, np.ndarray):
                    if x[i] <= bound[0]:
                        x[i] = bound[0]
                    elif x[i] >= bound[1]:
                        x[i] = bound[1]

            if feature_class == 'ordinal' or feature_class == 'ordinal_normalised':
                if not np.any(np.isclose(x[i], bound)):
                    val = bound[np.argmin(np.abs(np.array(bound) - x[i]))]
                    x[i] = val

        for group in self.categorical_groups:
            group_vals = x[group]
            closest_to_one = np.argmin(np.abs(group_vals - 1))
            group_vals.fill(0)
            group_vals[closest_to_one] = 1

            x[group] = group_vals 
        
        return x
                    
    def describe_instance(self, x: np.ndarray) -> None:
        """
        Print a human-readable description of a datapoint x, using feature names etc.. TODO: undo any preprocessing done by the dataset .describe_dataset() method.
        """

        self.check_valid_instance(x)

        for i in range(self.n_features):
            if self.feature_classes[i] != 'categorical':
                print(f"{self.feature_names[i]}, {x[i]}")

        for group in self.categorical_groups:
            index_active = group[np.where(x[group] == 1)[0][0]]
            print(f"categorical {self.feature_names[index_active]}")
   

    def __str__(self) -> str:
        """
        Output a description of the feature names, classes and bounds of this InputProperties instance
        """
         
        def format_row(row):
            return "| " + " | ".join(f"{str(item):<15}" for item in row) + " |"

        header = ["Index", "Feature Name", "Feature Class", "Bounds"]

        table = [header] + [[i, 
                             self.feature_names[i], 
                             self.feature_classes[i], 
                             self.bound[i],
                            ] for i in range(self.n_features)]
        
        table_str = "\n".join(format_row(row) for row in table)
        
        categorical_groups_str = "\n".join([f"Group {i+1}: {group}" for i, group in enumerate(self.categorical_groups)])
        if categorical_groups_str == "":
            categorical_groups_str = "None"
        
        return f"Input Properties:\n{table_str}\n\nCategorical Groups:\n{categorical_groups_str}"


    def gp_set_input_var_constraints(self, grb_model: gp.Model) -> Tuple[list[gp.Var], gp.MVar]:
        """
        Given a `grb_model`, output (equivalent) Input Vars and an Input MVar, of length n_features, which are constrained to be correctly encoded.
        """
         
        input_vars = []

        input_mvar = grb_model.addMVar(
            shape=(self.n_features,),
            vtype=GRB.CONTINUOUS,
            lb=-float('inf'),
            ub=float('inf'),
            name=f"inp_var"
        )
        
        for i, feature in enumerate(self.feature_names):
            feature_class = self.feature_classes[i]
            bound = self.bound[i]
            lb = -float('inf') 
            ub = float('inf') 

            if (feature_class == 'numeric' or feature_class == 'ordinal') and bound != None: 
                lb = bound[0]
                ub = bound[1]
            elif feature_class == 'categorical' or feature_class == 'ordinal_normalised':
                lb = 0
                ub = 1

            vtype = GRB.CONTINUOUS if feature_class == 'numeric' else GRB.INTEGER

            if feature_class == 'ordinal_normalised':
                chosen_value = grb_model.addMVar(shape=(len(bound),), vtype=GRB.BINARY, name=feature+"_picker")
                grb_model.addConstr(gp.quicksum(chosen_value) == 1)
                grb_model.addConstr(input_mvar[i] == gp.quicksum(bound[k] * chosen_value[k] for k in range(len(bound))))
                input_vars.append(chosen_value)

            else:
                var = grb_model.addVar(lb=lb, ub=ub, vtype=vtype, name=feature)
                grb_model.addConstr(input_mvar[i] == var)
                input_vars.append(var)

        for group in self.categorical_groups:
            group_vars = [input_vars[i] for i in group]
            grb_model.addConstr(gp.quicksum(group_vars) == 1)

        return input_vars, input_mvar
    


