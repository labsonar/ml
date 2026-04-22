"""Basic __init__.py
Allows to import the __all__ by folder name.
"""
from .time_processors import TimeProcessor, SampleProcessor, CPADetector, ToFloatConverter, SimpleProcessor
from .model_processors import VAEEncoder
from .spectral_processors import SpectralProcessor

__all__ = [
    "TimeProcessor",
    "SampleProcessor",
    "CPADetector",
    "ToFloatConverter",
    "SimpleProcessor",
    "VAEEncoder",
    "SpectralProcessor"
]
