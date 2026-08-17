#!/usr/bin/env python3

"""
Generate a complete reconstructed Iemanja dataset using one VAE model.

The generated dataset has the structure:

    output_root/
    └── model_name/
        ├── database.csv
        ├── ship_catalog.csv
        ├── acoutic_scenario_catalog.csv
        ├── dynamic_catalog.csv
        ├── data/
        │   ├── 0.wav
        │   ├── 1.wav
        │   └── ...
        └── comparisons/
            ├── 123_psd.png
            ├── 123_mel.png
            └── ...

The VAE is applied to the complete WAV before any fragmentation.

PSD and Mel comparisons are generated only for the validation set.
"""
#!/usr/bin/env python3

import os
import shutil
import argparse

import tqdm

import torch
import torchaudio

import lps_utils.utils as lps_utils
import lps_utils.quantities as lps_qty
import lps_ml.utils.device as ml_utils
import lps_ml.utils.general as ml_gen
import lps_ml.datasets.synthetic as ml_synthetic
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.acoustical.analysis as lps_analysis



def copy_dataset_metadata(input_dir: str, output_dir: str):
    """
    Copy the metadata CSV files required by Iemanja.
    """

    metadata_files = [
        "database.csv",
        "ship_catalog.csv",
        "acoutic_scenario_catalog.csv",
        "dynamic_catalog.csv",
    ]

    for filename in metadata_files:

        source = os.path.join(input_dir, filename)
        destination = os.path.join(output_dir, filename)

        if not os.path.exists(source):
            raise FileNotFoundError(
                f"Required dataset file not found: {source}"
            )

        shutil.copy2(source, destination)

        print(f"Copied: {filename}")

def get_validation_ids(
        input_dir: str,
        batch_size: int = 32,
        num_workers: int = 1,
):
    """
    Return the file IDs belonging to the validation split of Iemanja.
    """

    builder = ml_synthetic.IemanjaBuilder()

    parser = argparse.ArgumentParser(add_help=False)
    builder.add_argparse_args(parser)

    args = parser.parse_args([
        "--ie-dataset-dir", input_dir,
        "--ie-batch-size", str(batch_size),
        "--ie-num-workers", str(num_workers),
    ])

    dm = builder.from_argparse_args(args)
    dm.setup()

    if dm.val_df is None:
        raise RuntimeError("Validation dataframe was not initialized.")

    validation_ids = set(dm.val_df["file_id"].dropna().astype(int).unique().tolist())

    if not validation_ids:
        raise RuntimeError("Validation set contains no file IDs.")

    return validation_ids

def save_comparison_plots(
        original_data,
        reconstruction_data,
        fs,
        output_base,
):
    """
    Save PSD and Mel comparisons.
    """

    psd_file = output_base + "_psd.png"

    lps_bb.plot_psds(
        filename=psd_file,
        noises=[
            original_data,
            reconstruction_data
        ],
        labels=[
            "Original",
            "Reconstrução"
        ],
        window_size=4096,
        overlap=0.5,
        fs=lps_qty.Frequency.hz(fs)
    )

    mel_file = output_base + "_mel.png"

    lps_analysis.plot_spectral_analysis(
        filename=mel_file,
        signals=[
            original_data,
            reconstruction_data
        ],
        labels=[
            "Original",
            "Reconstrução"
        ],
        fs=lps_qty.Frequency.hz(fs),
        analysis=lps_analysis.SpectralAnalysis.MELGRAM,
        params=lps_analysis.Parameters(
            n_spectral_pts=4096,
            overlap=0.5,
            n_mels=512
        )
    )

def _main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate a complete reconstructed Iemanja dataset "
            "using one VAE model and generate validation-set "
            "PSD/Mel comparisons."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="TorchScript VAE model (.ts)"
    )

    parser.add_argument(
        "--compactness",
        type=int,
        default=1024,
        help=(
            "VAE compactness factor. The number of input samples "
            "is truncated to a multiple of this value."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="/data/iemanja",
        help=(
            "Original Iemanja dataset directory containing "
            "database.csv and data/"
        )
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/iemanja_vae",
        help="Root directory for reconstructed datasets."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used to determine the validation split."
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers used to determine the validation split."
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Model name
    # ------------------------------------------------------------------

    model_name = os.path.splitext(os.path.basename(args.model))[0]

    output_dir = os.path.join(args.output_dir, model_name)
    data_output_dir = os.path.join(output_dir, "data")
    comparison_dir = os.path.join(output_dir, "comparisons")

    os.makedirs(data_output_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("Iemanja VAE reconstruction")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Model name:  {model_name}")
    print(f"Input:       {args.input_dir}")
    print(f"Output:      {output_dir}")
    print("=" * 70)
    print()

    print("Loading VAE model...")

    device = ml_utils.get_available_device()

    print(f"Device: {device}")

    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    print()
    print("Determining validation set...")

    validation_ids = get_validation_ids(
        input_dir=args.input_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Validation files: {len(validation_ids)}")

    print()
    print("Copying dataset metadata...")

    copy_dataset_metadata(input_dir=args.input_dir, output_dir=output_dir)

    # ------------------------------------------------------------------
    # Find WAV files
    # ------------------------------------------------------------------

    input_data_dir = os.path.join(args.input_dir, "data")

    if not os.path.isdir(input_data_dir):
        raise FileNotFoundError(
            f"Dataset data directory not found: {input_data_dir}"
        )

    wav_files = lps_utils.find_files(input_data_dir)

    print()
    print(f"Found {len(wav_files)} WAV files.")

    reconstructed = 0
    comparisons = 0

    print()
    print("Reconstructing audio...")

    for wav_path in tqdm.tqdm(wav_files):

        waveform, fs = torchaudio.load(wav_path)
        n_samples = waveform.shape[-1]

        n_samples = n_samples // args.compactness * args.compactness
        waveform = waveform[..., :n_samples]

        with torch.inference_mode():
            reconstruction = model(waveform)

        original_data = waveform.detach().cpu().squeeze().numpy()
        recon_data = reconstruction.detach().cpu().squeeze().numpy()

        if recon_data.ndim == 2:
            recon_data = recon_data[0]

        filename = os.path.basename(wav_path)

        output_file = os.path.join(
            data_output_dir,
            filename
        )

        ml_gen.save_convert_wav(data=recon_data, fs=fs, filename=output_file)
        reconstructed += 1

        file_stem = os.path.splitext(filename)[0]

        try:
            file_id = int(file_stem)
        except ValueError:
            file_id = None

        if file_id is not None and file_id in validation_ids:

            comparison_base = os.path.join(comparison_dir, file_stem)

            save_comparison_plots(
                original_data=original_data,
                reconstruction_data=recon_data,
                fs=fs,
                output_base=comparison_base,
            )

            comparisons += 1

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)
    print(f"Reconstructed files : {reconstructed}")
    print(f"Validation files    : {len(validation_ids)}")
    print(f"Comparisons created : {comparisons}")
    print()
    print(f"Dataset             : {output_dir}")
    print(f"Audio               : {data_output_dir}")
    print(f"Comparisons         : {comparison_dir}")
    print("=" * 70)


if __name__ == "__main__":
    _main()
