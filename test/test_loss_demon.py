import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import wavfile

import lps_ml.model.audio_vae as lps_ml


def load_wav(path):
    fs, data = wavfile.read(path)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    else:
        data = data.astype(np.float32)

    return fs, data


def align_signals(x, y):
    """Corta para o menor tamanho"""
    T = min(x.shape[-1], y.shape[-1])
    return x[..., :T], y[..., :T]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav1", type=str)
    parser.add_argument("wav2", type=str)
    args = parser.parse_args()

    # load
    fs1, data1 = load_wav(args.wav1)
    fs2, data2 = load_wav(args.wav2)

    if fs1 != fs2:
        raise ValueError("Sample rates diferentes ainda não suportados.")

    # tensor
    x = torch.tensor(data1).unsqueeze(0).unsqueeze(0)
    y = torch.tensor(data2).unsqueeze(0).unsqueeze(0)

    # alinhar tempo
    x, y = align_signals(x, y)

    demon = lps_ml.DemonLoss(sample_rate=fs1)

    with torch.no_grad():
        X = demon.demon_spectrogram(x)[0]
        Y = demon.demon_spectrogram(y)[0]

    # garantir mesmo shape
    T = min(X.shape[-1], Y.shape[-1])
    X = X[:, :T]
    Y = Y[:, :T]

    # spectral convergence
    sc = torch.norm(X - Y) / (torch.norm(X) + 1e-7)

    # log magnitude
    log_X = torch.log(X + 1e-7)
    log_Y = torch.log(Y + 1e-7)
    log_mag = torch.mean(torch.abs(log_X - log_Y))

    print("sc: ", sc)
    print("log_mag: ", log_mag)

    # diferenças
    diff_mag = torch.abs(X - Y)
    diff_db = torch.abs(
        torch.log(X + 1e-7) - torch.log(Y + 1e-7)
    )

    mean_X = torch.mean(X, dim=-1).cpu().numpy()
    mean_Y = torch.mean(Y, dim=-1).cpu().numpy()
    mean_diff_mag = torch.mean(diff_mag, dim=-1).cpu().numpy()
    mean_diff_db = torch.mean(diff_db, dim=-1).cpu().numpy()

    # numpy
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()
    diff_mag = diff_mag.cpu().numpy()
    diff_db = diff_db.cpu().numpy()

    # saída
    base = os.path.splitext(args.wav1)[0]
    out_path = "./result/loss_comp"
    os.makedirs(out_path, exist_ok=True)

    # plot
    plt.figure(figsize=(12, 10))

    def plot_subplot(data, title, idx):
        plt.subplot(2, 2, idx)
        plt.imshow(
            (data + 1e-7),
            aspect='auto',
            origin='lower'
        )
        plt.title(title)
        plt.colorbar()

    plot_subplot(X, "DEMON - Sinal 1", 1)
    plot_subplot(Y, "DEMON - Sinal 2", 2)
    plot_subplot(diff_mag, "|X - Y| (Magnitude)", 3)
    plot_subplot(diff_db, "|log(X) - log(Y)| (dB diff)", 4)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "demon.png"), dpi=150)
    plt.close()

    print(f"Saved comparison image to: {out_path}")

    out_path_lines = base + "_demon_mean.png"

    plt.figure(figsize=(10, 6))

    freq_bins = np.arange(len(mean_X))

    plt.figure(figsize=(12, 10))

    def plot_subplot2(data, title, idx):
        plt.subplot(2, 2, idx)
        plt.plot(
            freq_bins,
            (data + 1e-7)
        )
        plt.title(title)

    plot_subplot2(mean_X, "DEMON - Sinal 1", 1)
    plot_subplot2(mean_Y, "DEMON - Sinal 2", 2)
    plot_subplot2(mean_diff_mag, "|X - Y| (Magnitude)", 3)
    plot_subplot2(mean_diff_db, "|log(X) - log(Y)| (dB diff)", 4)

    plt.xlabel("Frequency bins (DEMON)")
    plt.ylabel("Amplitude (dB)")
    plt.title("DEMON Mean Spectrum Comparison")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "demon_mean.png"), dpi=150)
    plt.close()

    print(f"Saved mean comparison image to: {out_path_lines}")

if __name__ == "__main__":
    main()