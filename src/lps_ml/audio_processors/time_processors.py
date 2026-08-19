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
        if fs == self.fs_out:
            return fs, data

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

class ToFloatConverter(ml_core.AudioPipeline):
    """AudioPipeline that converts int16 audio to float32 in range [-1, 1]."""

    def process(
        self,
        fs: lps_qty.Frequency,
        data: np.ndarray
    ) -> typing.Tuple[lps_qty.Frequency, np.ndarray]:

        data_float = data.astype(np.float32) / 2**15
        data_float = np.clip(data_float, -1.0, 1.0)
        return fs, data_float

class SampleProcessor(ml_core.AudioProcessor):
    """ Base processor that operates on the number of samples (last dimension). """

    def __init__(
        self,
        n_samples: int,
        overlap: int,
        fs_out: lps_qty.Frequency = None,
        audio_pipelines: typing.Sequence[ml_core.AudioPipeline] | None = None,
        sample_pipelines: typing.Sequence[ml_core.SamplePipeline] | None = None,
    ):
        audio_pipelines = audio_pipelines = list(audio_pipelines or [])

        if fs_out is not None:
            self.audio_pipelines.insert(0, Resampler(fs_out=fs_out))

        super().__init__(audio_pipelines, sample_pipelines)

        self.n_samples = n_samples
        self.overlap = overlap


    def fragment(self, fs: lps_qty.Frequency, data: np.ndarray) -> typing.List[np.ndarray]:

        data_samples = data.shape[-1]
        step = self.n_samples - self.overlap

        if step <= 0:
            raise ValueError("Overlap deve ser menor que n_samples.")

        windows = []

        for start in range(0, data_samples - self.n_samples + 1, step):
            slc = [slice(None)] * data.ndim
            slc[-1] = slice(start, start + self.n_samples)
            windows.append(data[tuple(slc)])

        return windows

class TimeProcessor(SampleProcessor):
    """Processor that defines sample windows in time."""

    def __init__(
        self,
        duration: lps_qty.Time,
        overlap: lps_qty.Time,
        fs_out: lps_qty.Frequency = None,
        audio_pipelines: typing.Sequence[ml_core.AudioPipeline] | None = None,
        sample_pipelines: typing.Sequence[ml_core.SamplePipeline] | None = None,
    ):
        super().__init__(
            n_samples=0,
            overlap=0,
            fs_out=fs_out,
            audio_pipelines=audio_pipelines,
            sample_pipelines=sample_pipelines,
        )

        self.duration = duration
        self.overlap_time = overlap

    def fragment(
        self,
        fs: lps_qty.Frequency,
        data: np.ndarray,
    ) -> typing.List[np.ndarray]:

        if (self.n_samples == 0):
            self.n_samples = int(self.duration * fs)
            self.overlap = int(self.overlap_time * fs)

        return super().fragment(fs=fs, data=data)

class SimpleProcessor(ml_core.AudioProcessor):
    """Processor that returns the complete processed audio as a single sample."""

    def __init__(
        self,
        audio_pipelines: typing.Sequence[ml_core.AudioPipeline] | None = None,
        sample_pipelines: typing.Sequence[ml_core.SamplePipeline] | None = None,
    ):
        super().__init__(
            audio_pipelines=audio_pipelines,
            sample_pipelines=sample_pipelines,
        )

    def fragment(
        self,
        fs: lps_qty.Frequency,
        data: np.ndarray,
    ) -> typing.List[np.ndarray]:

        return [data]
