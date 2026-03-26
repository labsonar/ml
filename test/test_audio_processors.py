import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchaudio

import lps_utils.quantities as lps_qty
import lps_ml.utils.sonar_loss as sonar_loss

def _main():

    parser = argparse.ArgumentParser(description="Test AudioProcessors")

    parser.add_argument(
        "wav_path",
        type=str,
        help="Caminho do arquivo .wav"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./result/audio_proc",
        help="Output folder for figures"
    )

    args = parser.parse_args()

    wav_path = args.wav_path
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)

    x, fs = torchaudio.load(wav_path)
    x = x.mean(dim=0, keepdim=True)
    x = x.unsqueeze(0)

    fs = lps_qty.Frequency.hz(fs)

    processors_2d = {
        "STFT": sonar_loss.STFT(sonar_loss.STFTConfig(n_fft=4096, hop_length=2048, temporal_integration=10)),
        "MEL": sonar_loss.Mel(sonar_loss.MelConfig(n_fft=4096, hop_length=2048, n_mels=256, sample_rate=fs, temporal_integration=10)),
        "LOFAR": sonar_loss.Lofar(sonar_loss.LofarConfig(n_fft=4096, hop_length=2048, temporal_integration=10)),
        "DEMON": sonar_loss.Demon(sonar_loss.DemonConfig(n_fft=512, hop_length=256, temporal_integration=10, decimate=[16, 16])),
    }

    processors_1d = {
        "STFT": sonar_loss.STFT(sonar_loss.STFTConfig(n_fft=4096, hop_length=2048,
            temporal_mean=True, temporal_integration=10)),
        "MEL": sonar_loss.Mel(sonar_loss.MelConfig(n_fft=4096, hop_length=2048, n_mels=256,
            temporal_mean=True, temporal_integration=10, sample_rate=fs)),
        "LOFAR": sonar_loss.Lofar(sonar_loss.LofarConfig(n_fft=4096, hop_length=2048,
            temporal_mean=True, temporal_integration=10)),
        "DEMON": sonar_loss.Demon(sonar_loss.DemonConfig(n_fft=512, hop_length=256,
            temporal_mean=True, temporal_integration=10, decimate=[16, 16])),
    }

    outputs_2d = {}
    outputs_1d = {}

    print("#######################")
    print("input: ", x.shape)

    with torch.no_grad():
        for name, proc in processors_2d.items():
            out = proc(x)  # (B, F, T)
            outputs_2d[name] = out.squeeze(0).cpu().numpy()
            print(name, " 2d: ", out.shape)

        for name, proc in processors_1d.items():
            out = proc(x)  # (B, F)
            outputs_1d[name] = out.squeeze(0).cpu().numpy()
            print(name, " 1d: ", out.shape)


    plt.figure(figsize=(12, 8))

    for i, (name, spec) in enumerate(outputs_2d.items(), 1):
        plt.subplot(2, 2, i)

        plt.imshow(
            spec,
            aspect="auto",
            origin="lower",
            cmap='jet'
        )
        plt.title(name)
        plt.xlabel("Tempo")
        plt.ylabel("Frequência")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "processors_2d.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(12, 8))

    plt.figure(figsize=(12, 8))

    for i, (name, vec) in enumerate(outputs_1d.items(), 1):
        plt.subplot(2, 2, i)

        plt.plot(vec)
        plt.title(f"{name} (mean)")
        plt.xlabel("Frequência")
        plt.ylabel("Magnitude (dB)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "processors_1d.png"), dpi=150)
    plt.close()

    print("Figuras salvas:")
    print(" - processors_2d.png")
    print(" - processors_1d.png")

if __name__ == "__main__":
    _main()