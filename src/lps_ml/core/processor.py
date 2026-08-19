"""Processor Module
"""
import abc
import typing

import numpy as np

import lps_utils.quantities as lps_qty
import lps_utils.hashable as utils_hash


class AudioPipeline(utils_hash.Hashable):
    """ Abstract class to process audios and get processed audios. """

    @abc.abstractmethod
    def process(self, fs: lps_qty.Frequency, data: np.array) \
            -> typing.Tuple[lps_qty.Frequency, np.array]:
        """
        Process an audio into processed audios.
        """

class SamplePipeline(utils_hash.Hashable):
    """ Abstract class to process samples and get processed samples. """

    @abc.abstractmethod
    def process(self, fs: lps_qty.Frequency, data: np.ndarray) -> \
            typing.Tuple[lps_qty.Frequency, np.ndarray]:
        """
        Process a single sample.
        """

class AudioProcessor(utils_hash.Hashable):
    """
    Base processor that transforms an audio into a list of samples.

    Processing is divided into three stages:
        audio_pipelines: operate on the complete audio.
        fragmentation: implemented by subclasses through ``fragment``.
        sample_pipelines: operate independently on each generated sample.
    """

    def __init__(
        self,
        audio_pipelines: typing.Sequence[AudioPipeline] | None = None,
        sample_pipelines: typing.Sequence[SamplePipeline] | None = None,
    ):
        super().__init__()
        self.audio_pipelines = list(audio_pipelines or [])
        self.sample_pipelines = list(sample_pipelines or [])

    @abc.abstractmethod
    def fragment(self, fs: lps_qty.Frequency, data: np.ndarray) -> typing.List[np.ndarray]:
        """ Fragment a processed audio into samples. """

    def process(self, fs: lps_qty.Frequency, data: np.ndarray) -> typing.List[np.ndarray]:
        """ Processed audio into processed samples. """

        for pipeline in self.audio_pipelines:
            fs, data = pipeline.process(fs=fs, data=data)

        samples = self.fragment(fs=fs, data=data)

        if not self.sample_pipelines:
            return samples

        processed_samples = []

        for sample in samples:
            for pipeline in self.sample_pipelines:
                fs, sample = pipeline.process(
                    fs=fs,
                    data=sample,
                )

            processed_samples.append(sample)

        return processed_samples
