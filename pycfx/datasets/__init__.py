"""
PyCFX Datasets

Datasets overview:
- `CaliforniaHousing` — Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters, 33 (1997) 291-297
- `GermanCredit` — Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.
- `GiveMeSomeCredit` - Credit Fusion and Will Cukierski. Give Me Some Credit. https://kaggle.com/competitions/GiveMeSomeCredit, 2011. Kaggle.
- `AdultIncome` — Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.
- Synthetic models
    - `SyntheticLinearlySeparable`
    - `SyntheticMulticlass`
    - `SyntheticMoons`
    - `SyntheticBimodal`

- Each dataset has corresponding metadata in an `InputProperties` object, and dimensionality reduction utilities are also included.
"""

from pycfx.datasets.datasets import *
from pycfx.datasets.dim_reduction import *
from pycfx.datasets.input_properties import InputProperties