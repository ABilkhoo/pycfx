"""
PyCFX Models: Wrappers for a variety of Python models for use in this library.

Models available:
- `PyTorchMLP`, MLP using PyTorch. For further customisation, you can subclass `PyTorchModel`
- `KerasMLP`, MLP using Keras. For further customisation, you can subclass `KerasModel`
- `RandomForestSKLearn`, `GradientBoostingSKLearn`, `DecisionTreeSKLearn`: SKLearn tree based classifiers
- Use the in-built training methods to train these models, or use `.load`/`.load_external` to point the wrapper to your pre-trained model file/object.
- Subclass `AbstractModel` to define your own model. See DifferentiableModel, MILPEncodableModel


"""

from pycfx.models.abstract_model import AbstractModel, MILPEncodableModel, DifferentiableModel
from pycfx.models.decisiontree_sklearn import DecisionTreeSKLearn
from pycfx.models.randomforest_sklearn import RandomForestSKLearn
from pycfx.models.gradientboosting_sklearn import GradientBoostingSKLearn
from pycfx.models.latent_encodings import *
from pycfx.models.mlp_keras import KerasMLP, KerasModel
from pycfx.models.mlp_pytorch import PyTorchMLP, PyTorchModel
