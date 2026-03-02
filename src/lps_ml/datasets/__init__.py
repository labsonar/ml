"""Basic __init__.py
Allows to import the __all__ by folder name.
"""
from .four_classes import FourClasses
from .iara import IARA
from .mnist import MNIST
from .synthetic import Iemanja
from .audio_folder import AudioFolder

__all__ = [
    "FourClasses",
    "IARA",
    "MNIST",
    "Iemanja",
    "AudioFolder"
]
