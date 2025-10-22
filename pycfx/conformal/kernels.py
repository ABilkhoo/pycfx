"""
pycfx/conformal/kernels.py
Kernel functions to be used by BaseLCP
"""

from pycfx.datasets.input_properties import InputProperties

import numpy as np
from typing import Callable

"""
Stores each kernel to be used by BaseLCP
"""
KERNEL_REGISTRY = {}

def register_kernel(key: str):
    """
    Decorator to apply to kernel function (of signature x1 (point), x2 (point), h (kernel bandwidth), input_properties: InputProperties) to add it to the registry.
    """
    def decorator(kernel_fn):
        KERNEL_REGISTRY[key] = kernel_fn
        return kernel_fn
    return decorator


def get_kernel(key: str) -> Callable[[np.float_, np.float_, np.float_, InputProperties], np.float_]:
    """
    Obtain a with key from the registry
    """
    return KERNEL_REGISTRY[key]

def feature_distance(x1, x2, ord, input_properties):
    """
    Compute a mixed feature distance, which is the difference between x1[i] and x2[i] for non-categorical features.
    For categorical features, take the distance to be 1 if they match and 0 otherwise.
    Return the norm of these feature distances, specified by ord.
    """
    if input_properties is None:
        return np.linalg.norm(x1 - x2, ord)

    dist_elements = np.zeros(input_properties.n_distinct_features)
    j = 0

    for i in range(input_properties.n_features):
        if input_properties.feature_classes[i] != 'categorical':
            dist_elements[j] += x1[i] - x2[i]
            j += 1

    for group in input_properties.categorical_groups:
        group_vals = x1[group]
        group_vals_2 = x2[group]
        dist_elements[j] = np.sum(group_vals == group_vals_2) 
        j += 1

    return np.linalg.norm(dist_elements, ord)

@register_kernel("gaussian")
def gaussian_kernel(x1, x2, h, input_properties=None):
    """Gaussian kernel exp(-||x1-x2||_2^2 / 2h^2)"""
    diff = feature_distance(x1, x2, 2, input_properties)
    return np.exp(-1 * diff * diff / (2 * h * h))

@register_kernel("box_l1")
def box_kernel_l1(x1, x2, h, input_properties=None):
    """Box kernel with L1 norm 1(||x1-x2||_1 <= h)"""
    dist = feature_distance(x1, x2, 1, input_properties)
    return int(dist <= h)

@register_kernel("box_l2")
def box_kernel_l2(x1, x2, h, input_properties=None):
    """Box kernel with L2 norm 1(||x1-x2||_2 <= h)"""
    dist = feature_distance(x1, x2, 2, input_properties)
    return int(dist <= h)

@register_kernel("box_linf")
def box_kernel_linf(x1, x2, h, input_properties=None):
    """Box kernel with Linf norm 1(||x1-x2||_inf <= h)"""
    dist = feature_distance(x1, x2, np.inf, input_properties)
    return int(np.max(dist) <= h)
