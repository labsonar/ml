import os
import typing
import math
import argparse
import torch
import torchaudio
import numpy as np
import torch.nn as nn

from scipy.signal import firwin, kaiserord
from scipy.optimize import fmin

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_sig
import lps_sp.acoustical.broadband as lps_bb
import lps_ml.utils.pqmf as ml_pqmf



# ============================================================
# UTILS
# ============================================================

def save_audio(tensor, fs, filename):

    tensor = torch.clamp(tensor, -1.0, 1.0)

    tensor_int16 = (tensor * 32767.0).to(torch.int16)

    signal_np = tensor_int16.detach().cpu().numpy()

    signal_np = np.squeeze(signal_np)

    lps_sig.save_wav(signal_np, fs, filename)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Test PQMF analysis/synthesis from RAVE"
    )

    parser.add_argument(
        "wav",
        type=str,
        help="Input wav file",
    )

    parser.add_argument(
        "--bands",
        type=int,
        default=16,
        help="Number of PQMF bands (default: 16)",
    )

    parser.add_argument(
        "--atten",
        type=int,
        default=100,
        help="Stopband attenuation in dB (default: 100)",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="./result/pqmf_test",
        help="Output directory",
    )

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    waveform, fs = torchaudio.load(args.wav)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    waveform = waveform / waveform.abs().max()

    waveform = waveform.unsqueeze(0)

    print("Input shape:", waveform.shape)

    pqmf = ml_pqmf.PQMF(attenuation=args.atten, n_band=args.bands)

    with torch.no_grad():

        subbands = pqmf(waveform)

        print("subbands shape:", subbands.shape)

        recon = pqmf.reverse(subbands)

    input_signal = waveform.squeeze().cpu().numpy()

    recon_signal = recon.squeeze().cpu().numpy()

    # --------------------------------------------------------
    # Save audio
    # --------------------------------------------------------

    save_audio(
        waveform.squeeze(),
        fs,
        os.path.join(args.out, "input.wav"),
    )

    save_audio(
        recon.squeeze(),
        fs,
        os.path.join(args.out, "reconstructed.wav"),
    )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    mse = np.mean((input_signal - recon_signal) ** 2)

    mae = np.mean(np.abs(input_signal - recon_signal))

    snr = 10 * np.log10(
        np.sum(input_signal ** 2)
        / np.sum((input_signal - recon_signal) ** 2)
    )

    print("\nErrors")
    print("MSE:", mse)
    print("MAE:", mae)
    print("SNR:", snr)

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    signals = [input_signal, recon_signal]

    labels = ["original", "reconstructed"]

    psd_filename = os.path.join(args.out, "psd.png")
    demon_filename = os.path.join(args.out, "demon.png")

    lps_bb.plot_psds(
        filename=psd_filename,
        noises=signals,
        labels=labels,
        fs=lps_qty.Frequency.hz(fs),
        window_size=1024 * 16,
        overlap=0.5,
    )

    lps_bb.plot_demon_lines(
        filename=demon_filename,
        signals=signals,
        labels=labels,
        fs=lps_qty.Frequency.hz(fs),
    )

if __name__ == "__main__":
    main()
