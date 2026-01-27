"""
Audio dataloader Module

This module provides functionality to read .wav files in a dataset.
"""
import os
import typing
import logging

import numpy as np

import scipy.io.wavfile as scipy_wav

import lps_utils.quantities as lps_qty
import lps_utils.hashable as utils_hash

class AudioFileLoader(utils_hash.Hashable):
    """ Class to find and load audio files """

    def __init__(self,
                 data_base_dir: str,
                 extract_id: typing.Callable[[str], int]):
        self.data_base_dir = data_base_dir
        self.extract_id = extract_id
        self.file_dict = {}
        self._find_raw_files()

    def _get_params(self):
        return {
            'data_base_dir': self.data_base_dir,
            'file_dict': len(self.file_dict)
        }

    def _find_raw_files(self) -> str:
        for root, _, files in os.walk(self.data_base_dir):
            for file in files:
                filename, extension = os.path.splitext(file)

                if extension.lower() != ".wav":
                    continue

                try:
                    file_id = self.extract_id(filename)
                except Exception as e: # pylint: disable=broad-exception-caught
                    logging.warning("Fail to extract ID from %s: %s", filename, e)
                    continue

                if isinstance(file_id, int):
                    self.file_dict[file_id] = os.path.join(root, file)
                else:
                    logging.warning("Invalid ID from %s: %s", filename, file_id)

    def load(self, file_id: int) -> typing.Tuple[lps_qty.Frequency, np.array]:
        """
        Function to load data from file

        Args:
            file_id (int): The ID to search for.

        Returns:
            typing.Tuple[
                lps_qty.Frequency:        sample frequency
                np.array:     data
            ]:
        """

        if file_id not in self.file_dict:
            raise UnboundLocalError(f'file {file_id} not found in {self.data_base_dir}')

        fs, data = scipy_wav.read(self.file_dict[file_id])

        if data.ndim != 1:
            data = data[:,0]

        return lps_qty.Frequency.hz(fs), data
