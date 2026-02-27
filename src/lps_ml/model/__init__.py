"""Basic __init__.py
Allows to import the __all__ by folder name.
"""
from .cnn import CNN
from .mlp import MLP
from .vae import VAE

__all__ = [
    "CNN",
    "MLP",
    "VAE"
]
