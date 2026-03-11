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



# ============================================================
# PQMF CORE (RAVE simplified)
# ============================================================

def reverse_half(x):
    mask = torch.ones_like(x)
    mask[..., 1::2, ::2] = -1
    return x * mask


def center_pad_next_pow_2(x):
    next_2 = 2 ** math.ceil(math.log2(x.shape[-1]))
    pad = next_2 - x.shape[-1]
    return nn.functional.pad(x, (pad // 2, pad // 2 + int(pad % 2)))


def get_qmf_bank(h, n_band):

    k = torch.arange(n_band).reshape(-1, 1)
    N = h.shape[-1]

    t = torch.arange(-(N // 2), N // 2 + 1)

    p = (-1) ** k * math.pi / 4

    mod = torch.cos((2 * k + 1) * math.pi / (2 * n_band) * t + p)

    hk = 2 * h * mod

    return hk


def kaiser_filter(wc, atten):

    N, beta = kaiserord(atten, wc / np.pi)
    N = 2 * (N // 2) + 1

    h = firwin(N, wc, window=('kaiser', beta), scale=False, nyq=np.pi)

    return h


def loss_wc(wc, atten, M):

    h = kaiser_filter(wc, atten)

    g = np.convolve(h, h[::-1], "full")

    g = abs(g[g.shape[-1] // 2::2 * M][1:])

    return np.max(g)


def get_prototype(atten, M):

    wc = fmin(lambda w: loss_wc(w, atten, M), 1 / M, disp=0)[0]

    return kaiser_filter(wc, atten)


class PQMF(nn.Module):

    def __init__(self, attenuation=100, n_band=16):

        super().__init__()

        h = get_prototype(attenuation, n_band)

        h = torch.from_numpy(h).float()

        hk = get_qmf_bank(h, n_band)

        hk = center_pad_next_pow_2(hk)

        self.register_buffer("hk", hk)

        self.n_band = n_band

    def analysis(self, x):

        x = nn.functional.conv1d(
            x,
            self.hk.unsqueeze(1),
            stride=self.n_band,
            padding=self.hk.shape[-1] // 2,
        )[..., :-1]

        return reverse_half(x)

    def synthesis(self, x):

        x = reverse_half(x)

        hk = self.hk.flip(-1)

        y = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.n_band * x.shape[-1],
        ).to(x)

        y[..., ::self.n_band] = x * self.n_band

        y = nn.functional.conv1d(
            y,
            hk.unsqueeze(0),
            padding=hk.shape[-1] // 2,
        )[..., 1:]

        return y


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

    pqmf = PQMF(attenuation=args.atten, n_band=args.bands)

    with torch.no_grad():

        subbands = pqmf.analysis(waveform)

        print("subbands shape:", subbands.shape)

        recon = pqmf.synthesis(subbands)

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