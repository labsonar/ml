"""
"""
import typing
import numpy as np

import torch

import lps_sp.acoustical.broadband as lps_bb
import lps_sp.acoustical.analysis as lps_analysis
import lps_sp.signal as lps_sig
import lps_utils.quantities as lps_qty


def _to_int16(tensor: torch.Tensor) -> np.ndarray:
    """
    Clamp a float waveform tensor to [-1, 1] and convert it to int16 PCM,
    as expected by scipy.io.wavfile / lps_sp.signal.save_wav.
    """
    tensor = torch.clamp(tensor, -1.0, 1.0)
    tensor_int16 = (tensor * 32767.0).to(torch.int16)
    signal_np = tensor_int16.detach().cpu().numpy()
    return np.squeeze(signal_np)

def _as_numpy_signal(x: typing.Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return _to_int16(x)
    return np.squeeze(np.asarray(x))

def save_comparison(
    signals: typing.List[np.ndarray],
    labels: typing.List[str],
    fs: typing.Union[int, lps_qty.Frequency],
    output_base: str,
    plots: typing.Sequence[str] = ("psd", "lofar", "mel", "demon"),
    window_size: int = 4096,
    overlap: float = 0.5,
) -> None:
    """
    Save a set of comparison plots (PSD / LOFAR / Mel / DEMON) for two or
    more aligned 1D signals (e.g. original vs one-or-more reconstructions).
    """
    if len(signals) != len(labels):
        raise ValueError("`signals` and `labels` must have the same length.")

    fs_qty = fs if isinstance(fs, lps_qty.Frequency) else lps_qty.Frequency.hz(fs)

    if "psd" in plots:
        lps_bb.plot_psds(
            filename=f"{output_base}_psd.png",
            noises=signals,
            labels=labels,
            fs=fs_qty,
            window_size=window_size,
            overlap=overlap,
        )

    if "demon" in plots:
        lps_bb.plot_demon_lines(
            filename=f"{output_base}_demon.png",
            signals=signals,
            labels=labels,
            fs=fs_qty,
        )

    if "lofar" in plots:
        lps_analysis.plot_spectral_analysis(
            filename=f"{output_base}_lofar.png",
            signals=signals,
            labels=labels,
            fs=fs_qty,
            analysis=lps_analysis.SpectralAnalysis.LOFAR,
        )

    if "mel" in plots:
        lps_analysis.plot_spectral_analysis(
            filename=f"{output_base}_mel.png",
            signals=signals,
            labels=labels,
            fs=fs_qty,
            analysis=lps_analysis.SpectralAnalysis.MELGRAM,
        )

def save_multi_signal_comparison(
    signals: typing.List[typing.Union[np.ndarray, torch.Tensor]],
    labels: typing.List[str],
    fs: typing.Union[int, "lps_qty.Frequency"],
    output_base: str,
    plots: typing.Sequence[str] = ("psd", "lofar", "mel", "demon"),
) -> None:
    """
    Saves each signal as "{output_base}_{label}.wav" and generates the comparison plots,
    """
    fs_qty = fs if isinstance(fs, lps_qty.Frequency) else lps_qty.Frequency.hz(fs)

    signals_np = []
    for signal, label in zip(signals, labels):
        signal_np = _as_numpy_signal(signal)
        signals_np.append(signal_np)

        safe_label = label.lower().replace(" ", "_")
        lps_sig.save_wav(signal_np, fs_qty, f"{output_base}_{safe_label}.wav")

    save_comparison(
        signals=signals_np,
        labels=labels,
        fs=fs_qty,
        output_base=output_base,
        plots=plots,
    )

def save_reconstruction_audio_and_comparison(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    fs: typing.Union[int, lps_qty.Frequency],
    output_base: str,
    original_label: str = "Original",
    reconstruction_label: str = "Reconstructed",
    plots: typing.Sequence[str] = ("psd", "lofar", "mel", "demon"),
) -> None:
    """
    Save both signals as .wav and generate the comparison plots.
    """

    save_multi_signal_comparison(
        signals=[original, reconstruction],
        labels=[original_label, reconstruction_label],
        fs=fs,
        output_base=output_base,
        plots=plots,
    )
