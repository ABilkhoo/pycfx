"""
pycfx/conformal/conformal_helpers.py
Helpers for localised conformal prediction classes
"""

from pycfx.datasets.dim_reduction import DimensionalityReduction
from pycfx.datasets.input_properties import InputProperties

import numpy as np

def sample_points(X: np.ndarray, y: np.ndarray, n_points: int=300, seed=0) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample n_points points from a dataset (X, y), and returned the sampled (X, y).
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(X), size=n_points, replace=False)
        return X[indices], y[indices]
    

def get_feature_ranges(input_properties: InputProperties, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each feature in dataset X, return the range provided by input_properties, or the observed range in X if input_properties does not specify a range.
    """
    feature_details = input_properties.get_feature_details()
    feature_ub = []
    feature_lb = []

    for i, feature in enumerate(feature_details):
        if feature[2][0] == float('-inf'):
            lb = np.min(X[:, i]) - 0.2*(np.max(X[:, i]) - np.min(X[:, i])) 
            feature_lb.append(lb)
        else:
            feature_lb.append(feature[2][0])

        if feature[2][1] == float('inf'):
            ub = np.max(X[:, i]) + 0.2*(np.max(X[:, i]) - np.min(X[:, i])) 
            feature_ub.append(ub)
        else:
            feature_ub.append(feature[2][1])

    return feature_lb, feature_ub


def generate_grid_points(input_properties: InputProperties, X: np.ndarray, dim_reduction: DimensionalityReduction = None, factor: int=1.5) -> np.array:
    """
    Sample points uniformly (i.e. from a grid) across the feature space.
    
    - input_properties: InputProperties for dataset
    - X: observed data to use to compute ranges if needed
    - dim_reduction: Provide a DimensionalityReduction object to sample the grid points from within a lower-dimensional space.
    - factor: Scale the number of returned points.

    Returns grid points.
    """
    if not dim_reduction:
        features_lb, features_ub = get_feature_ranges(input_properties, X)
    else:
        n_dims = dim_reduction.target_dim
        data_encoded = dim_reduction.encode(X)
        
        features_ub = []
        features_lb = []

        for i in range(n_dims):
            lb = np.min(data_encoded[:, i]) - 0.2*(np.max(data_encoded[:, i]) - np.min(data_encoded[:, i])) 
            features_lb.append(lb)

            ub = np.max(data_encoded[:, i]) + 0.2*(np.max(data_encoded[:, i]) - np.min(data_encoded[:, i])) 
            features_ub.append(ub)

    n_points = X.shape[0]
    grid_points = np.zeros_like(X)

    grid_points = np.meshgrid(
        *[np.linspace(features_lb[i], features_ub[i], num=int(factor*n_points**(1/len(features_lb)))) for i in range(len(features_lb))]
    )
    grid_points = np.array([point.flatten() for point in grid_points]).T

    return grid_points

def median_pairwise_distances(X: np.ndarray, dim_reduction: DimensionalityReduction = None) -> np.float_:
    """
    Compute the median of pairwise distances of points X (potentially after dimensionality reduction with dim_reduction)
    """
    pairwise_distances = []

    if dim_reduction:
        X = dim_reduction.encode(X)

    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            pairwise_distances.append(np.linalg.norm(X[i] - X[j]))

    return np.median(pairwise_distances)