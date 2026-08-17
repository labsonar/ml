#!/usr/bin/env python3

import os
import argparse
import tqdm

import torch
import torchaudio

import lps_utils.utils as lps_utils
import lps_utils.quantities as lps_qty
import lps_ml.utils.general as ml_gen
import lps_ml.audio_processors as ml_procs
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.acoustical.analysis as lps_analysis


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
    parser.add_argument("--compactness", type=int, default=1024)

    parser.add_argument(
        "--n_samples",
        type=int,
        default=None
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
    compactness = args.compactness

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

        if args.n_samples is not None and i >= args.n_samples:
            break

        waveform, fs = torchaudio.load(wav_path)

        #adjusting sample counts to multiples of the compaction factor
        n_samples = waveform.shape[-1]
        n_samples = (n_samples // compactness) * compactness
        waveform = waveform[..., :n_samples]

        waveform = waveform.unsqueeze(0)
        filename = os.path.basename(wav_path)

        for model_name, model in models.items():

            with torch.inference_mode():
                reconstruction = model(waveform)

            original_data = waveform.detach().cpu().squeeze().numpy()
            recon_data = reconstruction.detach().cpu().squeeze().numpy()

            if recon_data.ndim == 2:
                recon_data = recon_data[0]

            output_file = os.path.join(args.output_dir, model_name, filename)

            ml_gen.save_convert_wav(
                data=recon_data,
                fs=fs,
                filename=output_file
            )


            psd_file = os.path.splitext(output_file)[0] + "_psd.png"
            lps_bb.plot_psds(
                filename=psd_file,
                noises=[original_data, recon_data],
                labels=["Original", "Reconstrução"],
                window_size=4096,
                overlap=0.5,
                fs=lps_qty.Frequency.hz(fs)
            )

            psd_file = os.path.splitext(output_file)[0] + "_mel.png"
            lps_analysis.plot_spectral_analysis(
                filename=psd_file,
                signals=[original_data, recon_data],
                labels=["Original", "Reconstrução"],
                fs=lps_qty.Frequency.hz(fs),
                analysis=lps_analysis.SpectralAnalysis.MELGRAM,
                params=lps_analysis.Parameters(
                    n_spectral_pts=4096,
                    overlap=0.5,
                    n_mels=512
                )
            )


if __name__ == "__main__":
    main()
