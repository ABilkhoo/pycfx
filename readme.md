# PyCFX: Generating Counterfactual Explanations in Python

PyCFX is a library for generating counterfactual explanations for trained models in Python, and benchmarking counterfactual explanation generators.

With machine learning models being increasingly deployed in high-stakes scenarios affecting individuals, algorithmic recourse - providing actionable feedback to individuals influenced by these decisions - is crucial. For an individual `x` for which a given model `f` does not predict the desired class, e.g. the model rejects the individual from obtaining a loan given features such as income, age, etc., the a counterfactual explanation (CFX) provides a modified `x'` for which the model `f` does predict the target class, e.g. a generator may find the closest such `x'`. Further desiderata for CFXs include sparsity, diversity, causality, actionability, and plausibility.

PyCFX includes a range of CFX generators and has out-of-the-box support for a range of models and CFX generators. PyCFX is fully documented, built to be extendable by those wishing to benchmark their own CFX generator or use their own models or datasets, or use custom metrics to benchmark generators with. In comparison with existing libraries, PyCFX is built to include proper handling of numeric, categorical and ordinal features, has support for MILP (mixed-integer linear programming) based generators, and includes conformal prediction for uncertainty quantification.

See the example notebooks to get started! View the documentation here.
---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Datasets & Models](#datasets--models)
- [License](#license)

---

## Overview

PyCFX was developed for use by Aman Bilkhoo for use in the paper "CONFEX: Uncertainty-Aware Counterfactual Explanations with Conformal Gurantees" [1]
This library contains
- Sample synthetic and tabular datasets
- Model wrappers for PyTorch models, Keras models, SKLearn RandomForest, GradientBoosting and DecisionTree classifiers.
- Classes for split conformal prediction and localised conformal prediction.
- Counterfactual explanation generators (gradient-based, MILP-based, tree-based, plus more, some of which have leverage external implementations)
- Benchmarker utilities and evaluation metrics
- Visualisation tools
- Tutorial notebooks

For reporting bugs, issues and feature requests please open an issue.


## Requirements
- Python: 3.10
- See `requirements.txt` for dependencies.
- Install by cloning the repo, then quick start with the tutorial notebooks. The library will be made available via PyPi in a future release.

Note that for MILP-based generators make use of `gurobipy`, which requires a Gurobi licence to be available on your system. See [here](https://support.gurobi.com/hc/en-us/articles/12872879801105-How-do-I-retrieve-and-set-up-a-Gurobi-license) for further details.

## Quick start

Walk through tutorial notebooks at `tutorial/` and `results/`

## Project structure

```
pycfx/
├── pycfx/                                     
    ├── benchmarker/                         # counterfactual benchmarker, metrics, factories for generators and models
    ├── conformal/                           # conformal prediction wrapper over models, including split conformal prediction, localised CP, and localised CP via CONFEXTree
    ├── counterfactual_explanations/         # CFX generators including gradient-based generators, MILP-based generators, tree-based generators. Includes some external generators.
    ├── datasets/                            # dataset management, included synthetic and tabular datasets, input properties, dimensionality reduction
    ├── helpers/                             # helpers for constants and visualisation
    ├── library/                             # external code copied into this repo then modified, used for external generators or MILP encodings. See each file for origins.
    ├── models/                              # wrappers for a variety of Python models for use in this library.
├── results/                                 # scripts and notebooks detailing the use of the CFX benchmarker.
├── tutorials/                               # quick-start with some visual examples
```

See `results/readme.md` to reproduce the results from "CONFEX: Uncertainty-Aware Counterfactual Explanations with Conformal Gurantees"

## Datasets, Models and Generators.

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

Models available:
- `PyTorchMLP`, MLP using PyTorch. For further customisation, you can subclass `PyTorchModel`
- `KerasMLP`, MLP using Keras. For further customisation, you can subclass `KerasModel`
- `RandomForestSKLearn`, `GradientBoostingSKLearn`, `DecisionTreeSKLearn`: SKLearn tree based classifiers
- Use the in-built training methods to train these models, or use `.load`/`.load_external` to point the wrapper to your pre-trained model file/object.
- Subclass `AbstractModel` to define your own model

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

Conformal prediction overview:
- `SplitConformalPrediction`: Vanilla split conformal prediction
- `BaseLCP`: Localised conformal prediction (see [Guan, 2021](https://arxiv.org/abs/2106.08460))
- `ConformalCONFEXTree`: Localised CP via CONFEXTree, see [1]
- Registries for score functions and kernel functions
- `losses_conformal`: Smooth set size losses from Stutz et. al. "Learning optimal conformal classifiers." (2021), for use in gradient-based generators that utilise conformal prediction.

Benchmarking overview:
- `CFBenchmarker` allows you to specify a dataset, the scale of benchmark (number of test points), metrics to use, models to use and generators to benchmark. See usage in `results/`
- Produces results DataFrames for programmatic use of results, stores results in JSON and LATEX tables, and can produce figures.
- Available metrics: Validity, Failures, Implausibility, Plausibility (LOF), Distance, Sensitivity, Stability.

## Extensions

This library is being extended to include:
1. More generators, including REVISE (see WIP: branch ) and FACE.
2. More models and CFX generators that support those models (e.g. XGBoost models)
3. More localised conformal prediction methods e.g. CQC, SLCP, LoCart (see WIP)
4. Potential extension of the CONFEX method [1] to differentiable models (see WIP)
...

## License

This work is open-source under the Apache license. See LICENCE and NOTICE for more details.

