"""
pycfx/benchmarker/metrics.py
Metrics to the benchmarker CFBenchmarker to evaluate generated counterfactual explanations with.
"""

from pycfx.models.abstract_model import AbstractModel
from pycfx.datasets.input_properties import InputProperties
from pycfx.datasets.datasets import Dataset

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from scipy.special import softmax
from typing import List, Dict

class CFBenchmarkerMetric:
    """
    Abstract metric for use with CFBenchmarker
    """

    def get_factuals_bank(self, model: AbstractModel, input_properties: InputProperties, split_dataset: Dataset, factuals_bank: Dict[str, tuple[np.ndarray, np.ndarray]], seed: int) -> tuple[str, np.array, np.array]:
        """
        Specify additional factuals for which the benchmarker should compute CFXs for. 
        Use the `model`, `input_properties`, `split_dataset`, and current `factuals_bank` (consisting of a dictionary of bank name to (factuals, targets)), and a random seed.
        Return a bank name, bank factuals, and bank targets.
        """
        return None

    def compute_metric(self, model: AbstractModel, input_properties: InputProperties, dataset: Dataset, factuals_bank: Dict[str, tuple[np.ndarray, np.ndarray]], counterfactuals_bank: Dict[str, np.array]) -> List:
        """
        Get the metric computation (unaggregated) for this metric given the `model`, `input_properties`, `dataset`, `factuals_bank` and `counterfactuals_bank`.
        `factuals_bank` is a dictionary of bank name to (factuals, targets)
        `counterfactuals_bank` is a dictionary of bank name to computed counterfactuals.
        Returns the unaggregated metrics
        """
        pass

    def name(self) -> str:
        """
        Get the name of this metric
        """
        pass

    def get_bank(self, factuals_bank: Dict[str, tuple[np.ndarray, np.ndarray]], counterfactuals_bank: Dict[str, np.array], bank_name: str="main", drop_nan: bool=True, ensure_enc: bool=True, input_properties: InputProperties=None, model: AbstractModel=None):
        """
        Obtain a cleaned bank of (factuals, targets, counterfactuals)
        Specify drop_nan to exclude nan points
        Specify ensure_enc and input_properties to exclude incorrectly encoded points
        Specify ensure_enc and model to exclude failed counterfactuals (i.e. those which do not change the prediction to the target class)
        """
        x_factuals, y_targets = factuals_bank[bank_name]
        x_counterfactuals = counterfactuals_bank[bank_name]

        x_factuals = np.array(x_factuals)
        y_targets = np.array(y_targets)
        x_counterfactuals = np.array(x_counterfactuals)

        if ensure_enc:
            for i in range(x_counterfactuals.shape[0]):
                if input_properties:
                    x_counterfactuals[i] = input_properties.fix_encoding(x_counterfactuals[i])
                if model and np.argmax(model.predict(x_counterfactuals[i])) != y_targets[i]:
                    x_counterfactuals[i] == np.nan

        if drop_nan:
            nan_indices = np.where(~np.isnan(x_counterfactuals).any(axis=1))[0]
            x_factuals = x_factuals[nan_indices]
            y_targets = y_targets[nan_indices]
            x_counterfactuals = x_counterfactuals[nan_indices]


        return x_factuals, y_targets, x_counterfactuals

   
class ValidityMetric(CFBenchmarkerMetric):
    """
    ValidityMetric: Is the generated CFX valid i.e. the model predicts the target class for it?
    """

    def compute_metric(self, model: AbstractModel, input_properties: InputProperties, split_dataset: Dataset, factuals_bank: Dict[str, tuple[np.ndarray, np.ndarray]], counterfactuals_bank: Dict[str, np.array]) -> List:
        x_factuals, y_targets, x_counterfactuals = self.get_bank(factuals_bank, counterfactuals_bank, input_properties=input_properties)

        if len(x_counterfactuals) == 0:
            return np.array([np.nan])
        
        validities = np.zeros_like(y_targets)
        for i in range(x_counterfactuals.shape[0]):
            if np.argmax(model.predict(x_counterfactuals[i])) == y_targets[i]:
                validities[i] = 1

        return validities
    
    def name(self):
        return "Validity"
    
class FailuresMetric(CFBenchmarkerMetric):
    """
    FailuresMetric: Did the generator obtain a CFX, i.e. did not fail due to any error or MILP infeasibility issues.
    """

    def compute_metric(self, model: AbstractModel, input_properties: InputProperties, split_dataset: Dataset, factuals_bank: Dict[str, tuple[np.ndarray, np.ndarray]], counterfactuals_bank: Dict[str, np.array]) -> List:
        x_factuals, y_targets, x_counterfactuals = self.get_bank(factuals_bank, counterfactuals_bank, drop_nan=False)
        return np.isnan(x_counterfactuals)
    
    def name(self):
        return "Failures"


class ImplausibilityMetric(CFBenchmarkerMetric):
    """
    ImplausibilityMetric: Is the generated CFX plausible to the target class? Uses the average distance of the CFX to the closest `included_prop`% of points in the target class 
    For implausiblity, see Altmeyer, Patrick, et al. "Faithful model explanations through energy-constrained conformal counterfactuals." (2024)
    """
     
    def __init__(self, included_prop=0.1):
        self.included_prop = included_prop

    def compute_metric(self, model: AbstractModel, input_properties: InputProperties, dataset: Dataset, factuals_bank: Dict[str, tuple[np.ndarray, np.ndarray]], counterfactuals_bank: Dict[str, np.array]) -> List:
        x_factuals, y_targets, x_counterfactuals = self.get_bank(factuals_bank, counterfactuals_bank, input_properties=input_properties, model=model)
        
        impl_scores = np.zeros((x_counterfactuals.shape[0],))
        data_by_class = {}

        for y_target in np.unique(y_targets):
            data_ind_class_y = np.where(dataset.y == y_target)
            data_X_class_y = dataset.X[data_ind_class_y]
            data_by_class[int(y_target)] = data_X_class_y
            
        for i in range(x_counterfactuals.shape[0]):
            X_counterfactual = x_counterfactuals[i]
            y_target = y_targets[i]
            data_class_y = data_by_class[int(y_target)]

            distances = np.linalg.norm(data_class_y - X_counterfactual, axis=1, ord=2)
            distances = np.sort(distances)

            impl_scores[i] = np.mean(distances[:int(len(distances) * self.included_prop)])

        return impl_scores
    
    def name(self):
        return f"Implausibility_{self.included_prop}"
        

class LOFMetric(CFBenchmarkerMetric):
    """
    LOFMetric: Is the generated CFX plausible to the target class? Uses the the local outlier factory with novelty=True, specified `n_neighbours`, and specified stratification by target class
    """
     
    def __init__(self, n_neighbours=100, stratified=False):
        self.n_neighbours = n_neighbours
        self.stratified = stratified

    def compute_metric(self, model: AbstractModel, input_properties: InputProperties, dataset: Dataset, factuals_bank: Dict[str, tuple[np.ndarray, np.ndarray]], counterfactuals_bank: Dict[str, np.array]) -> List:
        x_factuals, y_targets, x_counterfactuals = self.get_bank(factuals_bank, counterfactuals_bank, input_properties=input_properties, model=model)

        lof_scores = np.zeros((x_counterfactuals.shape[0],))
        if len(x_factuals) == 0:
            return np.array([np.nan])

        if self.stratified:
            for target in input_properties.get_labels():
                lof = LocalOutlierFactor(n_neighbors=self.n_neighbours, novelty=True, n_jobs=-1)
                lof_X = dataset.X[dataset.y == target]
                lof.fit(lof_X)
                cfs_indicies = y_targets == target
                if np.sum(cfs_indicies) > 0:
                    lof_scores[cfs_indicies] = lof.predict(x_counterfactuals[cfs_indicies])
        else:
            lof = LocalOutlierFactor(n_neighbors=self.n_neighbours, novelty=True, n_jobs=-1)
            lof.fit(dataset.X)
            lof_scores = lof.predict(x_counterfactuals)
        
        return lof_scores
    
    def name(self):
        return f"LOF_{self.n_neighbours}{'S' if self.stratified else ''}"

class DistanceMetric(CFBenchmarkerMetric):
    """
    DistanceMetric: How far (costly) is the CFX from the target class?
    Uses the L-`norm` distance, optionally weighted by `dist_weight` or specify mad=True to use the median absolute deviation
    """

    def __init__(self, norm=1, mad=False, dist_weight=None):
        self.norm = norm
        self.mad = mad
        self.dist_weight = dist_weight

    def compute_metric(self, model: AbstractModel, input_properties: InputProperties, dataset: Dataset, factuals_bank: Dict[str, tuple[np.ndarray, np.ndarray]], counterfactuals_bank: Dict[str, np.array]) -> List:
        if self.mad:
            med = np.median(dataset.X_train, axis=0)
            mad = np.median(np.abs(dataset.X_train - med), axis=0)
            mad[mad == 0] = 1e-9 
            self.dist_weight = 1 / mad

        x_factuals, y_targets, x_counterfactuals = self.get_bank(factuals_bank, counterfactuals_bank, input_properties=input_properties, model=model)

        if len(x_counterfactuals) == 0:
            return np.array([np.nan])

        if self.dist_weight is None:
            return np.linalg.norm(x_counterfactuals - x_factuals, axis=1, ord=self.norm)
        else:
            return np.linalg.norm(self.dist_weight * (x_counterfactuals - x_factuals), axis=1, ord=self.norm)

    def name(self) -> str:
        if self.mad:
            return f"Distance_mad_l{self.norm}"
        else:
            return f"Distance_w{self.dist_weight}_l{self.norm}"

class SensitivityMetric(CFBenchmarkerMetric):
    """
    SensitivityMetric: How much a counterfactual explanation changes when the original instance x is perturbed within a small neighbourhood. 
    Formally, given an input x and its counterfactual x_c, we uniformly sample a perturbed instance x' from the L_2 ball centred around the factual, compute a new counterfactual $x'_c$. 
    Sensitivity is then defined as the relative deviation between the two counterfactuals, normalised by the cost of the initial counterfactual.
    """

    def __init__(self, n_sensitivity=20, n_neighbours=4, budget=0.05):
        self.n_sensitivity = n_sensitivity
        self.n_neighbours = n_neighbours
        self.budget = budget
    
    def get_factuals_bank(self, model: AbstractModel, input_properties: InputProperties, split_dataset: Dataset, factuals_bank: Dict[str, tuple[np.ndarray, np.ndarray]], seed: int):
        x_factuals, y_targets = factuals_bank['main']
        x_factuals, y_targets = np.array(x_factuals), np.array(y_targets)

        X_factual_sensitivity = np.zeros((self.n_sensitivity, 
                                        self.n_neighbours, 
                                        x_factuals.shape[1]))

        for i in range(self.n_sensitivity):
                sensitivity_base = x_factuals[i]
                X_factual_sensitivity[i] = split_dataset.sample_neighbours(sensitivity_base, 
                                                                    budget=self.budget, 
                                                                    n_samples=self.n_neighbours,
                                                                    seed=seed)

        X_factual_sensitivity = X_factual_sensitivity.reshape(X_factual_sensitivity.shape[0] * X_factual_sensitivity.shape[1], -1)
        y_targets = np.repeat(y_targets[:self.n_sensitivity], self.n_neighbours)

        return self.name(), X_factual_sensitivity, y_targets 

    def compute_metric(self, model: AbstractModel, input_properties: InputProperties, dataset: Dataset, factuals_bank: Dict[str, tuple[np.ndarray, np.ndarray]], counterfactuals_bank: Dict[str, np.array]) -> List:
        x_factuals, y_targets, x_counterfactuals = self.get_bank(factuals_bank, counterfactuals_bank, drop_nan=False)

        x_factuals_sens, y_targets_sens, x_counterfactuals_sens = self.get_bank(factuals_bank, counterfactuals_bank, bank_name=self.name(), drop_nan=False)
        
        x_counterfactuals_sens = x_counterfactuals_sens.reshape((self.n_sensitivity, self.n_neighbours, x_factuals.shape[1]))

        x_factuals_sens = x_factuals_sens.reshape((self.n_sensitivity, self.n_neighbours, x_factuals.shape[1]))

        sensitivities = np.zeros(x_counterfactuals_sens.shape[0])

        for i in range(x_counterfactuals_sens.shape[0]):
            original_factual = x_factuals[i]
            original_cf = x_counterfactuals[i]
            costs = np.zeros((x_counterfactuals_sens.shape[1],))

            for j in range(x_counterfactuals_sens.shape[1]):
                # neighbour_factual = x_factuals_sens[i][j]
                neighbour_cf = x_counterfactuals_sens[i][j]
                #TODO: Change to j?
                if np.isnan(neighbour_cf).any():
                    costs[j-1] = np.nan
                else:
                    costs[j-1] = np.linalg.norm(neighbour_cf - original_cf, ord=2) / np.linalg.norm(original_cf - original_factual, ord=2)     

            sensitivities[i] = np.nanmean(costs)

        return sensitivities

    def name(self):
        return f"Sensitivity{self.n_sensitivity},{self.n_neighbours},{self.budget}"
        

class StabilityMetric(CFBenchmarkerMetric):
    """
    StabilityMetric: Stability measures how consistent the counterfactual is under perturbations applied directly to the counterfactual itself. 
    The counterfactual x_c is perturbed within a budgeted neighbourhood and evaluate the variance in the model predictions across these perturbed samples.
    Adapted from Dutta et. al. Robust counterfactual explanations for tree-based ensembles (2022)
    """

    def __init__(self, n_neighbours=4, budget=0.05):
        self.n_neighbours = n_neighbours
        self.budget = budget


    def compute_metric(self, model: AbstractModel, input_properties: InputProperties, dataset: Dataset, factuals_bank: Dict[str, tuple[np.ndarray, np.ndarray]], counterfactuals_bank: Dict[str, np.array]) -> List:
        x_factuals, y_targets, x_counterfactuals = self.get_bank(factuals_bank, counterfactuals_bank)

        if len(x_counterfactuals) == 0:
            return np.array([np.nan])

        stabilities = np.zeros((x_counterfactuals.shape[0]))

        for i in range(x_counterfactuals.shape[0]):
            counterfactual = x_counterfactuals[i]

            neighbours = dataset.sample_neighbours(counterfactual, 
                                        budget=self.budget, 
                                        n_samples=self.n_neighbours,
                                        seed=0)

            preds = model.predict(neighbours)

            if not np.allclose(np.sum(preds, axis=1), 1):
                preds = softmax(preds, axis=1)
                
            target_probs = preds[np.arange(preds.shape[0]), (y_targets[i]).astype(int)]

            stabilities[i] = np.mean(target_probs) - np.std(target_probs)

        return stabilities
    
    def name(self):
        return f"Stability{self.n_neighbours},{self.budget}"
    

def define_counterfactual_targets(X_factual: np.ndarray, model_enc: AbstractModel, n_classes: int):
        """
        Helper function to give some factual points X_factual a random target from n_classes, for which the model_enc does not currently predict.
        """
        y_pred = model_enc.predict(X_factual)
        y_cf = np.zeros((y_pred.shape[0],))
        for i in range(y_pred.shape[0]):
                y_pred_i = y_pred[i]
                y_pred_cls = np.argmax(y_pred_i)
                y_cf_i = np.random.choice([cls for cls in range(n_classes) if cls != y_pred_cls])
                y_cf[i] = y_cf_i

        return y_cf