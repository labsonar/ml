"""Time processors
"""
import typing

import numpy as np

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_signal
import lps_ml.core as ml_core

class Resampler(ml_core.AudioPipeline):
    """ AudioPipeline to change sample frequency. """

    def __init__(self,
                 fs_out: lps_qty.Frequency):
        super().__init__()
        self.fs_out = fs_out

    def process(self, fs: lps_qty.Frequency, data: np.array) \
            -> typing.Tuple[lps_qty.Frequency, np.array]:

        decimated_signal = lps_signal.decimate(data, fs/self.fs_out)
        return self.fs_out, decimated_signal


class CPADetector(ml_core.AudioPipeline):
    """ AudioPipeline that detects the highest energy point (CPA) and cuts a centered window."""

    def __init__(self,
                 analysis_window: lps_qty.Time,
                 crop_window: lps_qty.Time):
        super().__init__()
        self.analysis_window = analysis_window
        self.crop_window = crop_window

    def process(self,
                fs: lps_qty.Frequency,
                data: np.ndarray) -> typing.Tuple[lps_qty.Frequency, np.ndarray]:

        if data.ndim > 2:
            raise ValueError(f"Input signal must have at 1 dimension, received: {data.ndim}D.")

        if data.ndim == 2:
            if 1 in data.shape:
                data = data.squeeze()
            else:
                raise ValueError(f"Input signal must have at 1 dimension, received: {data.ndim}D.")

        elif data.ndim < 1:
            raise ValueError(f"Input signal must have at 1 dimension, received: {data.ndim}D.")


        n_samples = len(data)
        n_analysis = int(self.analysis_window * fs)
        n_crop = int(self.crop_window * fs)

        if n_crop > n_samples:
            raise ValueError("Crop window is larger than the input signal.")

        if n_analysis > n_samples:
            raise ValueError("Analysis window is larger than the input signal.")

        step = n_analysis // 4
        energies = []
        starts = []

        for start in range(0, n_samples - n_analysis + 1, step):
            window = data[start:start + n_analysis]
            energy = np.sum(window ** 2)
            energies.append(energy)
            starts.append(start)

        max_idx = int(np.argmax(energies))
        cpa_start = starts[max_idx]
        cpa_center = cpa_start + n_analysis // 2

        half_crop = n_crop // 2
        crop_start = max(0, cpa_center - half_crop)
        crop_end = min(n_samples, crop_start + n_crop)

        if crop_end - crop_start < n_crop:
            crop_start = max(0, crop_end - n_crop)

        cropped_signal = data[crop_start:crop_end]

        return fs, cropped_signal

class TimeProcessor(ml_core.AudioProcessor):
    """ Simple time processor for resampling, pipelined, and sliding windowing. """

    def __init__(self,
                 duration: lps_qty.Time,
                 overlap: lps_qty.Time,
                 fs_out: lps_qty.Frequency,
                 pipelines: typing.Union[ml_core.AudioPipeline,
                                         typing.List[ml_core.AudioPipeline]] = None):
        super().__init__()
        self.duration = duration
        self.overlap = overlap

        if pipelines is None:
            self.pipelines = []
        elif isinstance(pipelines, ml_core.AudioPipeline):
            self.pipelines = [pipelines]
        else:
            self.pipelines = pipelines

        if fs_out is not None:
            self.pipelines.insert(0, Resampler(fs_out=fs_out))

    def process(self, fs: lps_qty.Frequency, data: np.array) -> typing.List[np.array]:

        for pipeline in self.pipelines:
            fs, data = pipeline.process(fs=fs, data=data)

        window_size = int(self.duration * fs)
        overlap_size = int(self.overlap * fs)
        step = int(window_size - overlap_size)

        windows = []
        for start in range(0, len(data) - window_size + 1, step):
            windows.append(data[start:start + window_size])

        return windows

class SampleProcessor(ml_core.AudioProcessor):
    """ Simple time processor for resampling, pipelined, and sliding windowing. """

    def __init__(self,
                 n_samples: int,
                 overlap: int,
                 fs_out: lps_qty.Frequency,
                 pipelines: typing.Union[ml_core.AudioPipeline,
                                         typing.List[ml_core.AudioPipeline]] = None):
        super().__init__()
        self.n_samples = n_samples
        self.overlap = overlap

        if pipelines is None:
            self.pipelines = []
        elif isinstance(pipelines, ml_core.AudioPipeline):
            self.pipelines = [pipelines]
        else:
            self.pipelines = pipelines

        if fs_out is not None:
            self.pipelines.insert(0, Resampler(fs_out=fs_out))

    def process(self, fs: lps_qty.Frequency, data: np.array) -> typing.List[np.array]:

        step = self.n_samples - self.overlap

        windows = []
        for start in range(0, len(data) - self.n_samples + 1, step):
            windows.append(data[start:start + self.n_samples])

        return windows

class ToFloatConverter(ml_core.AudioPipeline):
    """AudioPipeline that converts int16 audio to float32 in range [-1, 1]."""

    def process(
        self,
        fs: lps_qty.Frequency,
        data: np.ndarray
    ) -> typing.Tuple[lps_qty.Frequency, np.ndarray]:

        data_float = data.astype(np.float32) / 2**15
        data_float = np.clip(data_float, -1.0, 1.0)
        data_float = data_float[np.newaxis, :]
        return fs, data_float
