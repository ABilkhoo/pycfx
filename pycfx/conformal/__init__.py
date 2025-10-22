"""
PyCFX Conformal Prediction:

Conformal prediction overview:
- `SplitConformalPrediction`: Vanilla split conformal prediction
- `BaseLCP`: Localised conformal prediction (see [Guan, 2021](https://arxiv.org/abs/2106.08460))
- `ConformalCONFEXTree`: Localised CP via CONFEXTree, see [1]
- Registries for score functions and kernel functions
- `losses_conformal`: Smooth set size losses from Stutz et. al. "Learning optimal conformal classifiers." (2021), for use in gradient-based generators that utilise conformal prediction.

"""

from pycfx.conformal.conformal_helpers import *
from pycfx.conformal.localised_conformal_lcp import *
from pycfx.conformal.localised_conformal_tree import *
from pycfx.conformal.losses_conformal import *
from pycfx.conformal.milp_utils import *
from pycfx.conformal.split_conformal import *
