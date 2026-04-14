"""Basic __init__.py
Allows to import the __all__ by folder name.
"""
from .cv import CrossValidator
from .datamodule import BaseDataModule, AudioDataModule, ProcessedDataset, PairedProcessedDataset, PairedAudioDataModule
from .loader import AudioFileLoader
from .processor import AudioProcessor, AudioPipeline

__all__ = [
    "CrossValidator",
    "BaseDataModule",
    "AudioDataModule",
    "ProcessedDataset",
    "PairedProcessedDataset",
    "PairedAudioDataModule",
    "AudioFileLoader",
    "AudioProcessor",
    "AudioPipeline"
]
