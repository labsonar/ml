#!/usr/bin/env python3

"""
Evaluate VAE reconstruction quality using Frechet Audio Distance (FAD).

The evaluation compares:

    Dataset 1 = original
    Dataset 2 = VAE reconstructed

The evaluation is performed independently for:

    - validation
    - test

Both datasets are loaded using IemanjaBuilder.

The FAD is calculated in the embedding space of VGGish.

For each split:

    original fragments -> VGGish -> distribution 1
    reconstructed fragments -> VGGish -> distribution 2

    distribution 1 <-> distribution 2
                       |
                       v
                      FAD

The VAE operates on 8-second audio fragments. Therefore the FAD
evaluation is also performed on the 8-second temporal samples
returned by the Iemanja DataLoaders.

Outputs:

    output_dir/
    ├── fad.csv
    ├── validation/
    │   ├── original_embeddings.npy
    │   ├── reconstructed_embeddings.npy
    │   ├── bootstrap.csv
    │   └── histogram.png
    │
    └── test/
        ├── original_embeddings.npy
        ├── reconstructed_embeddings.npy
        ├── bootstrap.csv
        └── histogram.png
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.utils.data as torch_data

from tqdm import tqdm
from scipy import linalg

from frechet_audio_distance import FrechetAudioDistance

import lps_ml.datasets as ml_db
import lps_ml.utils.device as ml_device
import lps_ml.utils.general as ml_utils


# ----------------------------------------------------------------------
# FAD
# ----------------------------------------------------------------------

def calculate_fad(
        original_embeddings,
        reconstructed_embeddings,
):
    """
    Calculate Frechet Audio Distance between two embedding
    distributions.

    Args:
        original_embeddings:
            Array with shape [N, D].

        reconstructed_embeddings:
            Array with shape [M, D].

    Returns:
        FAD scalar.
    """

    mu_original = np.mean(
        original_embeddings,
        axis=0,
    )

    mu_reconstructed = np.mean(
        reconstructed_embeddings,
        axis=0,
    )

    sigma_original = np.cov(
        original_embeddings,
        rowvar=False,
    )

    sigma_reconstructed = np.cov(
        reconstructed_embeddings,
        rowvar=False,
    )

    diff = (
        mu_original
        -
        mu_reconstructed
    )

    covmean = linalg.sqrtm(
        sigma_original.dot(
            sigma_reconstructed
        )
    )

    if not np.isfinite(covmean).all():

        eps = 1e-6

        offset = np.eye(
            sigma_original.shape[0]
        ) * eps

        covmean = linalg.sqrtm(
            (
                sigma_original
                +
                offset
            ).dot(
                sigma_reconstructed
                +
                offset
            )
        )

    if np.iscomplexobj(covmean):

        if not np.allclose(
            np.diagonal(covmean).imag,
            0,
            atol=1e-3,
        ):
            raise ValueError(
                "Large imaginary component in "
                "covariance matrix square root."
            )

        covmean = covmean.real

    fad = (
        diff.dot(diff)
        +
        np.trace(sigma_original)
        +
        np.trace(sigma_reconstructed)
        -
        2.0 * np.trace(covmean)
    )

    return max(float(fad), 0.0)


# ----------------------------------------------------------------------
# DataLoader
# ----------------------------------------------------------------------

def extract_audio(
        dataloader,
):
    """
    Extract temporal audio samples from an Iemanja DataLoader.

    The Iemanja DataLoader is expected to return:

        x, y

    where x is:

        [B, T]

    or:

        [B, 1, T]

    Returns:

        list of numpy arrays, one array per audio fragment.
    """

    audio = []

    for x, y in tqdm(
            dataloader,
            desc="Extracting audio",
    ):

        if isinstance(x, list):

            raise RuntimeError(
                "FAD evaluation expects a non-paired "
                "Iemanja DataLoader."
            )

        x = x.detach().cpu()

        if x.ndim == 3:

            if x.shape[1] != 1:

                raise RuntimeError(
                    "Expected mono audio with shape "
                    "[B, 1, T]. Got: "
                    f"{tuple(x.shape)}"
                )

            x = x[:, 0]

        elif x.ndim != 2:

            raise RuntimeError(
                "Expected audio with shape [B, T] "
                "or [B, 1, T]. Got: "
                f"{tuple(x.shape)}"
            )

        for sample in x:

            sample = sample.numpy().astype(
                np.float32
            )

            sample = np.clip(
                sample,
                -1.0,
                1.0,
            )

            audio.append(sample)

    return audio

def split_embeddings(
        embeddings,
        seed,
):
    """
    Randomly split embeddings into two independent
    subsets.
    """

    rng = np.random.default_rng(seed)

    indices = rng.permutation(
        len(embeddings)
    )

    midpoint = len(indices) // 2

    idx_a = indices[:midpoint]
    idx_b = indices[midpoint:]

    return (
        embeddings[idx_a],
        embeddings[idx_b],
    )
# ----------------------------------------------------------------------
# VGGish
# ----------------------------------------------------------------------

def compute_embeddings(
        fad_model,
        audio,
        sample_rate,
):
    """
    Compute VGGish embeddings for a list of
    temporal audio samples.
    """

    embeddings = fad_model.get_embeddings(
        audio,
        sr=sample_rate,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float64,
    )

    if embeddings.ndim != 2:

        raise RuntimeError(
            "Expected embeddings with shape [N, D]. "
            f"Got {embeddings.shape}"
        )

    return embeddings


# ----------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------
def bootstrap_fad(
        embeddings_a,
        embeddings_b,
        n_bootstrap,
        seed,
):
    """
    Bootstrap the FAD between two embedding distributions.

    The two distributions are resampled independently.

    This is appropriate for FAD because FAD compares
    distributions rather than paired samples.
    """

    rng = np.random.default_rng(seed)

    n_a = len(embeddings_a)
    n_b = len(embeddings_b)

    scores = []

    for _ in tqdm(
            range(n_bootstrap),
            desc="Bootstrap FAD",
    ):

        idx_a = rng.integers(
            0,
            n_a,
            size=n_a,
        )

        idx_b = rng.integers(
            0,
            n_b,
            size=n_b,
        )

        fad = calculate_fad(
            embeddings_a[idx_a],
            embeddings_b[idx_b],
        )

        scores.append(fad)

    return np.asarray(
        scores,
        dtype=np.float64,
    )

def summarize_bootstrap(scores):
    """
    Summarize bootstrap FAD distribution.
    """

    return {
        "mean": np.mean(scores),
        "variance": np.var(
            scores,
            ddof=1,
        ),
        "std": np.std(
            scores,
            ddof=1,
        ),
        "ci_low": np.percentile(
            scores,
            2.5,
        ),
        "ci_high": np.percentile(
            scores,
            97.5,
        ),
    }

def bootstrap_fad_same_distribution(
        embeddings,
        n_bootstrap,
        seed,
):
    """
    Bootstrap FAD between two independent samples
    drawn from the same empirical distribution.

    This provides the real-real FAD baseline.
    """

    rng = np.random.default_rng(seed)

    n = len(embeddings)

    scores = []

    for _ in tqdm(
            range(n_bootstrap),
            desc="Bootstrap real-real FAD",
    ):

        idx_a = rng.integers(
            0,
            n,
            size=n,
        )

        idx_b = rng.integers(
            0,
            n,
            size=n,
        )

        fad = calculate_fad(
            embeddings[idx_a],
            embeddings[idx_b],
        )

        scores.append(fad)

    return np.asarray(
        scores,
        dtype=np.float64,
    )
# ----------------------------------------------------------------------
# Histogram
# ----------------------------------------------------------------------

# def save_histogram(
#         scores,
#         filename,
#         split,
# ):
#     """
#     Save bootstrap FAD histogram.
#     """

#     fig, ax = plt.subplots(
#         figsize=(8, 6)
#     )

#     ax.hist(
#         scores,
#         bins=30,
#     )

#     mean = np.mean(scores)

#     ax.axvline(
#         mean,
#         linestyle="--",
#         linewidth=2,
#         label=f"mean = {mean:.4f}",
#     )

#     ax.set_xlabel(
#         "FAD"
#     )

#     ax.set_ylabel(
#         "Frequency"
#     )

#     ax.set_title(
#         f"VGGish FAD bootstrap - {split}"
#     )

#     ax.legend()

#     fig.tight_layout()

#     fig.savefig(
#         filename,
#         dpi=150,
#     )

#     plt.close(fig)

def save_histogram(
        baseline_scores,
        vae_scores,
        filename,
        split,
):
    """
    Save bootstrap FAD histograms for:

        REAL x REAL
        REAL x VAE
    """

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.hist(
        baseline_scores,
        bins=30,
        alpha=0.6,
        label="Real × Real",
    )

    ax.hist(
        vae_scores,
        bins=30,
        alpha=0.6,
        label="Real × VAE",
    )

    baseline_mean = np.mean(
        baseline_scores
    )

    vae_mean = np.mean(
        vae_scores
    )

    ax.axvline(
        baseline_mean,
        linestyle="--",
        linewidth=2,
        label=(
            f"Real × Real mean = "
            f"{baseline_mean:.4f}"
        ),
    )

    ax.axvline(
        vae_mean,
        linestyle="--",
        linewidth=2,
        label=(
            f"Real × VAE mean = "
            f"{vae_mean:.4f}"
        ),
    )

    ax.set_xlabel(
        "FAD"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_title(
        f"VGGish FAD bootstrap - {split}"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=150,
    )

    plt.close(fig)

# ----------------------------------------------------------------------
# Split evaluation
# ----------------------------------------------------------------------

def evaluate_split(
        original_loader,
        reconstructed_loader,
        fad_model,
        split,
        output_dir,
        sample_rate,
        n_bootstrap,
        seed,
):
    """
    Evaluate one Iemanja split.
    """

    split_dir = os.path.join(
        output_dir,
        split,
    )

    os.makedirs(
        split_dir,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(f"FAD evaluation - {split}")
    print("=" * 70)

    # --------------------------------------------------------------
    # Original
    # --------------------------------------------------------------

    print(
        f"\nExtracting original {split}..."
    )

    original_audio = extract_audio(
        original_loader
    )

    print(
        f"Original samples: "
        f"{len(original_audio)}"
    )

    # --------------------------------------------------------------
    # Reconstructed
    # --------------------------------------------------------------

    print(
        f"\nExtracting reconstructed {split}..."
    )

    reconstructed_audio = extract_audio(
        reconstructed_loader
    )

    print(
        f"Reconstructed samples: "
        f"{len(reconstructed_audio)}"
    )

    if len(original_audio) != len(
            reconstructed_audio
    ):

        raise RuntimeError(
            f"Different number of samples in "
            f"{split}: "
            f"original={len(original_audio)}, "
            f"reconstructed="
            f"{len(reconstructed_audio)}"
        )

    # --------------------------------------------------------------
    # VGGish embeddings
    # --------------------------------------------------------------

    print(
        f"\nComputing VGGish embeddings "
        f"for original {split}..."
    )

    original_embeddings = compute_embeddings(
        fad_model=fad_model,
        audio=original_audio,
        sample_rate=sample_rate,
    )

    print(
        "Original embeddings:",
        original_embeddings.shape,
    )

    print(
        f"\nComputing VGGish embeddings "
        f"for reconstructed {split}..."
    )

    reconstructed_embeddings = compute_embeddings(
        fad_model=fad_model,
        audio=reconstructed_audio,
        sample_rate=sample_rate,
    )

    print(
        "Reconstructed embeddings:",
        reconstructed_embeddings.shape,
    )

    # --------------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------------

    np.save(
        os.path.join(
            split_dir,
            "original_embeddings.npy",
        ),
        original_embeddings,
    )

    np.save(
        os.path.join(
            split_dir,
            "reconstructed_embeddings.npy",
        ),
        reconstructed_embeddings,
    )

    # --------------------------------------------------------------
    # FAD
    # --------------------------------------------------------------

    fad = calculate_fad(
        original_embeddings,
        reconstructed_embeddings,
    )

    print()
    print(
        f"{split.upper()} FAD: {fad:.6f}"
    )

    # --------------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------------
    # --------------------------------------------------------------
    # Bootstrap baseline: REAL x REAL
    # --------------------------------------------------------------

    baseline_bootstrap = bootstrap_fad_same_distribution(
        embeddings=original_embeddings,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    baseline_mean = np.mean(
        baseline_bootstrap
    )

    baseline_variance = np.var(
        baseline_bootstrap,
        ddof=1,
    )

    baseline_std = np.std(
        baseline_bootstrap,
        ddof=1,
    )

    baseline_ci_low, baseline_ci_high = np.percentile(
        baseline_bootstrap,
        [2.5, 97.5],
    )

    # --------------------------------------------------------------
    # Bootstrap: REAL x VAE
    # --------------------------------------------------------------

    bootstrap_scores = bootstrap_fad(
        embeddings_a=original_embeddings,
        embeddings_b=reconstructed_embeddings,
        n_bootstrap=n_bootstrap,
        seed=seed + 1,
    )

    bootstrap_mean = np.mean(
        bootstrap_scores
    )

    bootstrap_variance = np.var(
        bootstrap_scores,
        ddof=1,
    )

    bootstrap_std = np.std(
        bootstrap_scores,
        ddof=1,
    )

    ci_low, ci_high = np.percentile(
        bootstrap_scores,
        [2.5, 97.5],
    )

    print()
    print(
        f"{split.upper()} bootstrap"
    )

    print()
    print("REAL x REAL baseline")

    print(
        f"Mean     : {baseline_mean:.6f}"
    )

    print(
        f"Variance : {baseline_variance:.6e}"
    )

    print(
        f"Std      : {baseline_std:.6f}"
    )

    print(
        f"95% CI   : "
        f"[{baseline_ci_low:.6f}, "
        f"{baseline_ci_high:.6f}]"
    )

    print()
    print("REAL x VAE")

    print(
        f"Mean     : {bootstrap_mean:.6f}"
    )

    print(
        f"Variance : {bootstrap_variance:.6e}"
    )

    print(
        f"Std      : {bootstrap_std:.6f}"
    )

    print(
        f"95% CI   : "
        f"[{ci_low:.6f}, {ci_high:.6f}]"
    )

    # --------------------------------------------------------------
    # Bootstrap CSV
    # --------------------------------------------------------------

    bootstrap_df = pd.DataFrame(
        {
            "bootstrap": np.arange(
                len(bootstrap_scores)
            ),
            "fad_real_real": baseline_bootstrap,
            "fad_real_vae": bootstrap_scores,
        }
    )

    bootstrap_df.to_csv(
        os.path.join(
            split_dir,
            "bootstrap.csv",
        ),
        index=False,
    )

    # --------------------------------------------------------------
    # Histogram
    # --------------------------------------------------------------

    # save_histogram(
    #     scores=bootstrap_scores,
    #     filename=os.path.join(
    #         split_dir,
    #         "histogram.png",
    #     ),
    #     split=split,
    # )

    save_histogram(
        baseline_scores=baseline_bootstrap,
        vae_scores=bootstrap_scores,
        filename=os.path.join(
            split_dir,
            "histogram.png",
        ),
        split=split,
    )

    return {
        "split": split,

        "n_original": len(
            original_audio
        ),

        "n_reconstructed": len(
            reconstructed_audio
        ),

        # ----------------------------------------------------------
        # Direct FAD
        # ----------------------------------------------------------

        "fad_real_vae": fad,

        # ----------------------------------------------------------
        # Real × Real baseline
        # ----------------------------------------------------------

        "fad_real_real": baseline_mean,

        "baseline_variance": baseline_variance,

        "baseline_std": baseline_std,

        "baseline_ci_95_low":
            baseline_ci_low,

        "baseline_ci_95_high":
            baseline_ci_high,

        # ----------------------------------------------------------
        # Real × VAE bootstrap
        # ----------------------------------------------------------

        "fad_real_vae_bootstrap_mean":
            bootstrap_mean,

        "fad_real_vae_bootstrap_variance":
            bootstrap_variance,

        "fad_real_vae_bootstrap_std":
            bootstrap_std,

        "fad_real_vae_ci_95_low":
            ci_low,

        "fad_real_vae_ci_95_high":
            ci_high,

        # ----------------------------------------f------------------
        # Relative distance from baseline
        # ----------------------------------------------------------

        "fad_ratio":
            bootstrap_mean / baseline_mean,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VAE reconstruction quality "
            "using VGGish Frechet Audio Distance."
        )
    )

    parser.add_argument(
        "--dataset1-dir",
        type=str,
        required=True,
        help="Original Iemanja dataset.",
    )

    parser.add_argument(
        "--dataset2-dir",
        type=str,
        required=True,
        help="VAE reconstructed Iemanja dataset.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./result/fad",
        help="Output directory.",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate.",
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap FAD estimates.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    # --------------------------------------------------------------
    # Iemanja arguments
    # --------------------------------------------------------------

    builder = ml_db.IemanjaBuilder()

    builder.add_argparse_args(
        parser
    )

    args = parser.parse_args()

    torch.set_float32_matmul_precision(
        "medium"
    )

    ml_utils.set_seed()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    device = ml_device.get_available_device()

    print()
    print("=" * 70)
    print("Iemanja VAE FAD evaluation")
    print("=" * 70)

    print(
        f"Original dataset : "
        f"{args.dataset1_dir}"
    )

    print(
        f"VAE dataset      : "
        f"{args.dataset2_dir}"
    )

    print(
        f"Sample rate      : "
        f"{args.sample_rate}"
    )

    print(
        f"Bootstrap         : "
        f"{args.n_bootstrap}"
    )

    print(
        f"Device            : "
        f"{device}"
    )

    print(
        f"Output            : "
        f"{args.output_dir}"
    )

    print("=" * 70)

    # --------------------------------------------------------------
    # Build original DataModule
    # --------------------------------------------------------------

    print()
    print("Building original Iemanja DataModule...")

    args.ie_dataset_dir = (
        args.dataset1_dir
    )

    dm_original = (
        builder.from_argparse_args(args)
    )

    dm_original.setup()

    # --------------------------------------------------------------
    # Build reconstructed DataModule
    # --------------------------------------------------------------

    print()
    print(
        "Building reconstructed Iemanja "
        "DataModule..."
    )

    args.ie_dataset_dir = (
        args.dataset2_dir
    )

    dm_reconstructed = (
        builder.from_argparse_args(args)
    )

    dm_reconstructed.setup()

    # --------------------------------------------------------------
    # Verify split parity
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("Checking dataset split parity")
    print("=" * 70)

    for split in ["val", "test"]:

        if split == "val":

            df_original = (
                dm_original.val_df
            )

            df_reconstructed = (
                dm_reconstructed.val_df
            )

        else:

            df_original = (
                dm_original.test_df
            )

            df_reconstructed = (
                dm_reconstructed.test_df
            )

        ids_original = set(
            df_original[
                "file_id"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        ids_reconstructed = set(
            df_reconstructed[
                "file_id"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        print(
            f"{split:5s}: "
            f"original={len(ids_original):5d} | "
            f"reconstructed="
            f"{len(ids_reconstructed):5d}"
        )

        if ids_original != ids_reconstructed:

            raise RuntimeError(
                f"Split '{split}' is not identical "
                "between original and reconstructed "
                "datasets."
            )

    print(
        "Validation and test splits are identical."
    )

    print("=" * 70)

    # --------------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------------

    original_val_loader = (
        dm_original.val_dataloader(
            shuffle=False
        )
    )

    reconstructed_val_loader = (
        dm_reconstructed.val_dataloader(
            shuffle=False
        )
    )

    original_test_loader = (
        dm_original.test_dataloader(
            shuffle=False
        )
    )

    reconstructed_test_loader = (
        dm_reconstructed.test_dataloader(
            shuffle=False
        )
    )

    # --------------------------------------------------------------
    # VGGish
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("Loading VGGish")
    print("=" * 70)

    fad_model = FrechetAudioDistance(
        model_name="vggish",
        sample_rate=args.sample_rate,
        use_pca=False,
        use_activation=False,
        verbose=True,
    )

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    validation_result = evaluate_split(
        original_loader=original_val_loader,
        reconstructed_loader=reconstructed_val_loader,
        fad_model=fad_model,
        split="validation",
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    # --------------------------------------------------------------
    # Test
    # --------------------------------------------------------------

    test_result = evaluate_split(
        original_loader=original_test_loader,
        reconstructed_loader=reconstructed_test_loader,
        fad_model=fad_model,
        split="test",
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    results_df = pd.DataFrame(
        [
            validation_result,
            test_result,
        ]
    )

    results_df.to_csv(
        os.path.join(
            args.output_dir,
            "fad.csv",
        ),
        index=False,
    )

    print()
    print("=" * 70)
    print("FINAL FAD RESULTS")
    print("=" * 70)

    print(
        results_df[
            [
                "split",
                "n_original",
                "n_reconstructed",

                # ------------------------------------------------------
                # Direct FAD
                # ------------------------------------------------------

                "fad_real_vae",

                # ------------------------------------------------------
                # Real x Real baseline
                # ------------------------------------------------------

                "fad_real_real",
                "baseline_std",
                "baseline_ci_95_low",
                "baseline_ci_95_high",

                # ------------------------------------------------------
                # Real x VAE bootstrap
                # ------------------------------------------------------

                "fad_real_vae_bootstrap_mean",
                "fad_real_vae_bootstrap_std",
                "fad_real_vae_ci_95_low",
                "fad_real_vae_ci_95_high",

                # ------------------------------------------------------
                # Relative distance
                # ------------------------------------------------------

                "fad_ratio",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("Output")
    print("=" * 70)

    print(
        f"Summary   : "
        f"{args.output_dir}/fad.csv"
    )

    print(
        f"Validation: "
        f"{args.output_dir}/validation/"
    )

    print(
        f"Test      : "
        f"{args.output_dir}/test/"
    )

    print("=" * 70)


if __name__ == "__main__":
    _main()
