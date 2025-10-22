"""
pycfx/conformal/conformal_benchmarker.py
Benchmarking utilities for evaluating coverage and set size of a calibrated SplitConformalPrediction instance.
"""

from pycfx.conformal.split_conformal import SplitConformalPrediction

import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_stats(alpha: np.float_, pred_intervals: np.ndarray, y_correct: np.ndarray, indices: np.ndarray=None, cov_gap: bool=False) -> tuple[float, float]:
        """
        Compute set size and coverage for given prediction intervals and target classes.

        Parameters:
           - alpha: target miscoverage rate
           - pred_intervals: An list containing prediction intervals returned by conformal.predict_batch
           - y_correct: A list of the same length as pred_intervals, containing the true label
           - indices: Set to an array of index positions of to filter pred_intervals and y_correct by, or set to None for no filtering
           - cov_gap: If set to True, will return the percentage point difference between the target coverage rate and empirical coverage. 
                      Otherwise will return empirical empirical coverage rate as as a float.
        
        Returns:
        """

        if indices is not None:
            pred_intervals = [pred_intervals[i] for i in indices]
            y_correct = y_correct[indices]

        coverage = 0
        set_size = 0
        num_points = len(pred_intervals)

        if num_points == 0:
            return np.nan, np.nan

        for i in range(num_points):
            set_size += len(pred_intervals[i])
            if y_correct[i] in pred_intervals[i]:
                coverage += 1

        coverage /= num_points
        set_size /= num_points

        if cov_gap:
            coverage = coverage - (1 - alpha)
            coverage *= 100

        return set_size, coverage
    
def compute_stats_partition(alpha: np.float_, set_sizes: list, coverage: list, partition_sizes: list, only_penalise_undercoverage: bool=False, cov_gap: bool=False) -> tuple[float, float]:
    """
        Given a partitioned scenario test set, each providing an average set size and coverage rate, compute an overall set size and coverage gap

        Parameters:
           - alpha: target miscoverage rate
           - set_sizes: list of average set sizes for each partition
           - coverage: list of empirical coverage (float) for each partition
           - partition_sizes: list of lengths of each partition. 
           - only_penalise_undercoverage: when computing the coverage gaps, ignore overcoverage.
           - cov_gap: If set to True, will return the percentage point difference between the target coverage rate and empirical coverage. 
                      Otherwise will return empirical empirical coverage rate as as a float.
        
        Returns:
            - Average set size and average coverage gap across all partitions.
    """
    
    num_partitions = len(coverage)
    avg_set_size = np.sum((np.array(set_sizes) * np.array(partition_sizes))) / np.sum(partition_sizes)

    if not cov_gap:
        coverage = np.average(np.array(coverage))
        return avg_set_size, coverage
    
    cov_gaps = np.array(coverage) - (1 - alpha)

    if only_penalise_undercoverage:
        cov_gaps = np.minimum([0] * len(cov_gaps), cov_gaps)
        c_gap = np.sum(np.abs(cov_gaps)) * 100 / num_partitions
    else:
        c_gap = np.sum(cov_gaps) * 100 / num_partitions

    avg_set_size = np.sum((np.array(set_sizes) * np.array(partition_sizes))) / np.sum(partition_sizes)

    return avg_set_size, c_gap


def evaluate_conditional(conformal: SplitConformalPrediction, X: np.ndarray, y: np.ndarray, n_bins=10, seed=2, cov_gap=False):
    """
        Run a suite of tests for a calibrated SplitConformalPrediction instance
        Computes empirical coverage and set sizes:
        1. Across full test set (marginal)
        2. Across the test set, stratified by class (class-conditional)
        3. Across the test set, split randomly into equal-sized bins (random binning)
        4. Using a counterfactual simulation (for each element in the test set, find the nearest neighbour element in the test set which is predicted to be a singleton set in the opposite class). Use this resampling of the test set for computing empirical coverage and set sizes.

        Parameters:
           - conformal: Calibrated SplitConformalPrediction instance to test
           - X: Test data
           - y: Test labels
           - n_bins: number of bins to use for random binning experiment, default 10
           - cov_gap: If set to True, will return the percentage point difference between the target coverage rate and empirical coverage. 
                      Otherwise will return empirical empirical coverage rate as as a float.
           - seed: seed to use for random binning experiment
           - only_penalise_undercoverage: when computing the coverage gaps, ignore overcoverage.
           - cov_gap: If set to True, will return the percentage point difference between the target coverage rate and empirical coverage. 
                      Otherwise will return empirical empirical coverage rate as as a float.
        
        Returns:
            - Average set size and average coverage gap for experiments (1-4)
    """

    assert conformal.is_calibrated
    alpha = conformal.alpha
    pred_intervals = conformal.predict_batch(X)

    # print("Marginal")
    set_size, coverage = compute_stats(alpha, pred_intervals, y, cov_gap=cov_gap)
    # print(set_size, coverage)
    
    # print("Class conditional")
    unique_y = np.unique(y)
    set_sizes_cc = []
    coverages_cc = []
    partition_sizes_cc = []

    for label in unique_y:
        indices = np.where(y == label)[0]
        set_size, coverage = compute_stats(alpha, pred_intervals, y, indices)

        set_sizes_cc.append(set_size)
        coverages_cc.append(coverage)

        partition_sizes_cc.append(len(indices))

    # print(set_sizes_cc, coverages_cc, partition_sizes_cc)
    set_size_cc, cov_gap_cc = compute_stats_partition(alpha, set_sizes_cc, coverages_cc, partition_sizes_cc, cov_gap=cov_gap)
    # print(set_size_cc, cov_gap_cc)

    # print("Random binning")
    rng = np.random.default_rng(seed=seed)
    random_indices = np.array_split(rng.permutation(len(X)), n_bins)
    set_sizes_rb = []
    coverages_rb = []
    partition_sizes_rb = []

    for indices in random_indices:
        set_size, coverage = compute_stats(alpha, pred_intervals, y, indices)

        set_sizes_rb.append(set_size)
        coverages_rb.append(coverage)
        partition_sizes_rb.append(len(indices))

    # print(set_sizes_rb, coverages_rb, partition_sizes_rb)
    set_size_rb, cov_gap_rb = compute_stats_partition(alpha, set_sizes_rb, coverages_rb, partition_sizes_rb, cov_gap=cov_gap)
    # print(set_size_rb, cov_gap_rb)
    

    # print("Counterfactual simulation")
    
    y_targets = (y + 1) % conformal.input_properties.n_targets
    indicies_to_include = np.empty((X.shape[0],), dtype=np.int32)
    indicies_to_include[:] = -1

    singleton_points = {}
    for label in conformal.input_properties.get_labels():
        singleton_points[label] = np.array([i for i, interval in enumerate(pred_intervals) if interval == [label]])

    for label, indices in singleton_points.items():
        if len(indices) > 0:
            nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X[indices])
            singleton_points[label] = (indices, nn_model)
        else:
            singleton_points[label] = ([], None)

    for i, x in enumerate(X):
        if y_targets[i] in singleton_points:
            indices, nn_model = singleton_points[y_targets[i]]
            if len(indices) > 0:
                _, nearest_idx = nn_model.kneighbors([x], n_neighbors=1)
                indicies_to_include[i] = indices[nearest_idx[0][0]]

    indicies_to_include = indicies_to_include[indicies_to_include != -1]
    set_size_cf, coverage_cf = compute_stats(alpha, pred_intervals, y, cov_gap=cov_gap, indices=indicies_to_include)
    # print(set_size_cf, coverage_cf)

    return set_size, coverage, set_size_cc, cov_gap_cc, set_size_rb, cov_gap_rb, set_size_cf, coverage_cf

