"""
PyCFX Counterfactual Explanations

Generators available:
- Differentiable
    - [Wachter](https://arxiv.org/pdf/1711.00399)
    - [ECCCo](https://arxiv.org/abs/2312.10648)
    - [Schut](https://arxiv.org/abs/2103.08951), 
    - Support for PyTorch/Keras models.
- MILP-based
    - Min Distance CFX, 
    - CONFEX CFXs [1]
    - Support for PyTorch/Keras MLP and tree-based models
- Other: 
    - Nearest-neighbour CFX,
    - Support for all models.
- External: FOCUS (uses [CFXplorer](https://github.com/kyosek/CFXplorer) for implementation), FeatureTweak (implementation used from https://github.com/upura/featureTweakPy/blob/master/featureTweakPy.py), with support for SKLearn tree based classifiers.

"""

from pycfx.counterfactual_explanations.cf_conformal import ConformalCF
from pycfx.counterfactual_explanations.cf_generator import CounterfactualGenerator
from pycfx.counterfactual_explanations.cf_mindist import MinDistanceCF
from pycfx.counterfactual_explanations.cf_nearestneighbour import NearestNeighbourCF
from pycfx.counterfactual_explanations.cf_gradient_based import *
from pycfx.counterfactual_explanations.external.cf_featuretweak import *
from pycfx.counterfactual_explanations.external.cf_focus import *