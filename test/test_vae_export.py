#!/usr/bin/env python3

import os
import argparse
import tqdm

import torch
import torchaudio

import lps_utils.utils as lps_utils
import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_sig
import lps_ml.utils.general as ml_gen
import lps_sp.acoustical.broadband as lps_bb


def main():

    parser = argparse.ArgumentParser(
        description="Exporta reconstruções de WAV usando múltiplos autoencoders"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Lista de modelos TorchScript (.ts)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./result/vae_export"
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Diretório contendo arquivos WAV"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = lps_utils.find_files(args.input_dir)

    models = {}

    print("Loading models...")

    for model_name, model_path in ml_gen.shortest_relative_path(args.models):

        print(f"  {model_name}")

        model = torch.jit.load(model_path)
        model.eval()

        models[model_name] = model

        os.makedirs(
            os.path.join(args.output_dir, model_name),
            exist_ok=True
        )

    for i, wav_path in enumerate(tqdm.tqdm(files)):

        waveform, fs = torchaudio.load(wav_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(
                waveform,
                dim=0,
                keepdim=True
            )

        waveform = waveform / (
            waveform.abs().max() + 1e-12
        )

        waveform = waveform.unsqueeze(0)

        filename = os.path.basename(wav_path)

        for model_name, model in models.items():

            with torch.inference_mode():
                reconstruction = model(waveform)

            recon_data = reconstruction.detach().cpu().squeeze()

            if recon_data.ndim == 2:
                recon_data = recon_data[0]

            output_file = os.path.join(
                args.output_dir,
                model_name,
                filename
            )

            ml_gen.save_convert_wav(
                data=recon_data,
                fs=fs,
                filename=output_file
            )

            psd_file = os.path.splitext(output_file)[0] + ".png"
            lps_bb.plot_psds(
                filename=psd_file,
                noises=[waveform, recon_data],
                labels=["Original", "Reconstrução"],
                window_size=4096,
                overlap=0.5,
                fs=lps_qty.Frequency.hz(fs)
            )


if __name__ == "__main__":
    main()
