"""
pycfx/models/latent_encodings.py
Latent encodings for use in optimisation loops
"""


from abc import ABC, abstractmethod
import numpy as np

class LatentEncoding(ABC):
    """
    Abstract latent encoding
    """

    @abstractmethod
    def encode(self, x: np.ndarray) -> np.array:
        """
        Convert an input tensor to an encoded tensor
        """
        pass
    
    @abstractmethod
    def decode(self, z: np.ndarray) -> np.array:
        """
        Convert an encoded tensor to a tensor in the input space
        """
        pass

class IdentityEncoding(LatentEncoding):
    """
    Identity encoding: no change
    """
    def encode(self, x: np.ndarray) -> np.array:
        return x
    
    def decode(self, z: np.ndarray) -> np.array:
        return z