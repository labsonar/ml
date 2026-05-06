import os
import sys
import argparse
import tqdm
import numpy as np

import torch
import torchaudio

import lps_utils.quantities as lps_qty
import lps_utils.utils as lps_utils
import lps_sp.signal as lps_sig
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.acoustical.analysis as lps_analysis
import lps_ml.utils.sonar_loss as ml_loss

OUTPUT_DIR = "./result/rave_export"
os.makedirs(OUTPUT_DIR, exist_ok=True)


MODELS = [
    ("v0_80", "/data/models/v0_80.ts"),
    ("v0_95", "/data/models/v0_95.ts"),
    ("v0_995", "/data/models/v0_995.ts"),
    ("v0_999", "/data/models/v0_999.ts"),
    ("v0_99", "/data/models/v0_99.ts"),
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

    n_samples=int(2**17)    #8.192s

    parser = argparse.ArgumentParser(
        description="Compara reconstrução de WAV usando RAVE"
    )

    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument(
        "wavfile_directory",
        type=str,
        help="Diretório contendo arquivos WAV"
    )

    args = parser.parse_args()
    files = lps_utils.find_files(args.wavfile_directory)

    models = []
    for model_id, model_path in MODELS:

        model = torch.jit.load(model_path)
        model.eval()
        models.append((model_id, model))

        os.makedirs(os.path.join(OUTPUT_DIR, str(model_id)), exist_ok=True)

    lofar_dir = os.path.join(OUTPUT_DIR, "analysis", "lofar")
    demon_dir = os.path.join(OUTPUT_DIR, "analysis", "demon")
    psd_dir = os.path.join(OUTPUT_DIR, "analysis", "psd")
    os.makedirs(lofar_dir, exist_ok=True)
    os.makedirs(demon_dir, exist_ok=True)
    os.makedirs(psd_dir, exist_ok=True)

    max_files = args.max_files

    loss = ml_loss.SonarLoss()
    loss_accumulator = {
        model_id: {
            "stft": [],
            "mel": [],
            "lofar": [],
            "demon": [],
            "total": []
        }
        for model_id, _ in models
    }

    for i, wav_path in enumerate(tqdm.tqdm(files)):

        if max_files is not None and i >= max_files:
            break

        waveform, fs = torchaudio.load(wav_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform[:, :n_samples]
        waveform = waveform / waveform.abs().max()
        waveform = waveform.unsqueeze(0)

        signals = []
        labels = []

        in_data = waveform.detach().cpu().squeeze()

        signals.append(in_data.numpy())
        labels.append("Input")

        name = os.path.splitext(os.path.basename(wav_path))[0]
        for model_id, model in models:

            with torch.inference_mode():
                recon = model(waveform)

            recon_data = recon.detach().cpu().squeeze()

            if recon_data.ndim == 2:
                recon_data = recon_data[0, :]

            out_filename = os.path.join(OUTPUT_DIR, str(model_id), f"{name}.wav")

            save_audio(recon_data, fs, out_filename)

            signals.append(recon_data.numpy())
            labels.append(model_id)

            losses_dict = loss.compute_all_losses(waveform, recon)
            total_loss = loss(waveform, recon)

            loss_accumulator[model_id]["stft"].append(
                torch.stack(losses_dict["stft_loss"]).mean().item()
            )
            loss_accumulator[model_id]["mel"].append(
                torch.stack(losses_dict["mel_loss"]).mean().item()
            )
            loss_accumulator[model_id]["lofar"].append(
                torch.stack(losses_dict["lofar_loss"]).mean().item()
            )
            loss_accumulator[model_id]["demon"].append(
                torch.stack(losses_dict["demon_loss"]).mean().item()
            )
            loss_accumulator[model_id]["total"].append(
                total_loss.item()
            )

        psd_filename = os.path.join(psd_dir, f"{name}.png")
        demon_filename = os.path.join(demon_dir, f"{name}.png")
        lofar_filename = os.path.join(lofar_dir, f"{name}.png")

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

    print("\n===== RESULTADOS MÉDIOS =====\n")

    for model_id in loss_accumulator:

        print(f"Modelo: {model_id}")

        for key, values in loss_accumulator[model_id].items():
            if values:
                mean_val = sum(values) / len(values)
                print(f"  {key:>6}: {mean_val:.6f}")

        print()

if __name__ == "__main__":
    main()