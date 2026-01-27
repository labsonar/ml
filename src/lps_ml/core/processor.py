"""Processor Module
"""
import abc
import typing

import numpy as np

import lps_utils.quantities as lps_qty
import lps_utils.hashable as utils_hash

class AudioProcessor(utils_hash.Hashable):
    """ Abstract class to process audios and get processed windows. """

    @abc.abstractmethod
    def process(self, fs: lps_qty.Frequency, data: np.array) -> typing.List[np.array]:
        """
        Process an audio into processed windows.

        Args:
            fs (lps_qty.Frequency):  Sampling frequency of the input signal.
            data (np.array): Audio as a 1D tensor.

        Returns:
            window_list (list of np.array):  A list of processed windows
        """

class AudioPipeline(utils_hash.Hashable):
    """ Abstract class to process audios and get processed audios. """

    @abc.abstractmethod
    def process(self, fs: lps_qty.Frequency, data: np.array) \
            -> typing.Tuple[lps_qty.Frequency, np.array]:
        """
        Process an audio into processed audios.
        """
