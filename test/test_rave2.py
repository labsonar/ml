import os
import sys
import torch
import torchaudio
import numpy as np

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_sig
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.acoustical.analysis as lps_analysis

OUTPUT_DIR = "./result/rave_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)


MODELS = [
    ("v1", "/data/models/16k_v1.ts"),
    ("v4", "/data/models/16k_v4.ts"),
    ("v6", "/data/models/16k_v6.ts"),
]


def save_audio(tensor, fs, filename):

    tensor = torch.clamp(tensor, -1.0, 1.0)
    tensor_int16 = (tensor * 32767.0).to(torch.int16)

    signal_np = tensor_int16.detach().cpu().numpy()
    signal_np = np.squeeze(signal_np)

    if signal_np.ndim == 2:
        signal_np = signal_np.T

    lps_sig.save_wav(signal_np, fs, filename)


def main():

    if len(sys.argv) < 2:
        print("Uso: python test_rave_compare.py arquivo.wav")
        sys.exit(1)

    wav_path = sys.argv[1]

    waveform, fs = torchaudio.load(wav_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform / waveform.abs().max()
    waveform = waveform.unsqueeze(0)

    print("Input shape:", waveform.shape)

    signals = []
    labels = []

    # -------------------------
    # Input
    # -------------------------

    in_data = waveform.detach().cpu().squeeze()

    in_filename = os.path.join(OUTPUT_DIR, "input.wav")
    save_audio(in_data, fs, in_filename)

    signals.append(in_data.numpy())
    labels.append("Input")

    # -------------------------
    # Model loop
    # -------------------------

    for model_id, model_path in MODELS:

        print(f"Loading model: {model_id}")

        model = torch.jit.load(model_path)
        model.eval()

        with torch.inference_mode():
            recon = model(waveform)

        recon_data = recon.detach().cpu().squeeze()

        if recon_data.ndim == 2:
            recon_data = recon_data[0, :]

        out_filename = os.path.join(OUTPUT_DIR, f"{model_id}.wav")

        save_audio(recon_data, fs, out_filename)

        signals.append(recon_data.numpy())
        labels.append(model_id)

    # -------------------------
    # Plots
    # -------------------------

    psd_filename = os.path.join(OUTPUT_DIR, "psd.png")
    demon_filename = os.path.join(OUTPUT_DIR, "demon.png")
    lofar_filename = os.path.join(OUTPUT_DIR, "lofar.png")

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

    lps_analysis.plot_spectral_analysis(
        filename=lofar_filename,
        signals=signals,
        labels=labels,
        fs=lps_qty.Frequency.hz(fs),
    )


if __name__ == "__main__":
    main()