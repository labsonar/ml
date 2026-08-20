#!/usr/bin/env python3

"""
Evaluate generative quality using a SVDD embedding.

The SVDD model is trained on dataset 1 (reference/original).

The evaluation compares:

    Dataset 1 = real/reference
    Dataset 2 = synthetic/reconstructed

The following metrics are calculated in the SVDD embedding space:

    - alpha-precision
    - beta-recall
    - authenticity

The evaluation is performed independently for validation and test sets.

Expected output:

    output_dir/
    ├── metrics.csv
    ├── curves_val.csv
    └── curves_test.csv
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

import lps_ml.datasets as ml_db
import lps_ml.model.svdd as ml_svdd
import lps_ml.utils.device as ml_device
import lps_ml.utils.general as ml_utils


def load_model(path: str):
    """
    Load an SVDD model from a checkpoint.

    If a directory is provided, best.ckpt is used.
    """

    if os.path.isdir(path):
        path = os.path.join(path, "best.ckpt")

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    print(f"Loading SVDD model: {path}")

    model = ml_svdd.SVDDMLP.load_from_checkpoint(path)
    model.eval()

    return model


def build_datamodule(
    builder,
    args,
    dataset_dir: str,
):
    """
    Build an Iemanja DataModule.
    """

    args.ie_dataset_dir = dataset_dir

    dm = builder.from_argparse_args(args)
    dm.setup()

    return dm


def get_split_ids(
    dm,
    split: str,
):
    """
    Return file IDs belonging to a split.
    """

    if split == "train":
        df = dm.train_df

    elif split == "val":
        df = dm.val_df

    elif split == "test":
        df = dm.test_df

    else:
        raise ValueError(
            f"Invalid split: {split}"
        )

    return set(
        df["file_id"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )


def verify_split_parity(
    dm1,
    dm2,
):
    """
    Verify that the two datasets have identical
    train/validation/test file IDs.
    """

    print()
    print("=" * 70)
    print("Checking dataset split parity")
    print("=" * 70)

    for split in ["train", "val", "test"]:

        ids1 = get_split_ids(dm1, split)
        ids2 = get_split_ids(dm2, split)

        only_1 = ids1 - ids2
        only_2 = ids2 - ids1

        print(
            f"{split:5s}: "
            f"dataset1={len(ids1):5d} | "
            f"dataset2={len(ids2):5d}"
        )

        if only_1 or only_2:

            print(
                f"\nERROR: split '{split}' is not identical."
            )

            if only_1:
                print(
                    f"  Only dataset1: {len(only_1)} files"
                )

            if only_2:
                print(
                    f"  Only dataset2: {len(only_2)} files"
                )

            raise RuntimeError(
                "Dataset splits are not paired. "
                "Evaluation cannot be safely performed."
            )

    print()
    print("All splits have identical file IDs.")
    print("=" * 70)


def evaluate_split(
    model,
    real_dataloader,
    synthetic_dataloader,
):
    """
    Calculate generative metrics for one dataset split.
    """

    metrics = model.calculate_alpha_beta_authenticity(
        real_dataloader=real_dataloader,
        synthetic_dataloader=synthetic_dataloader,
    )

    return metrics


def save_curves(
    metrics: dict,
    filename: str,
    title: str,
):
    """
    Save alpha-precision and beta-recall curves.
    """

    alphas = metrics["alphas"]

    alpha_precision = (
        metrics["alpha_precision_curve"]
    )

    beta_recall = (
        metrics["beta_recall_curve"]
    )

    curves_df = pd.DataFrame(
        {
            "alpha": alphas,
            "alpha_precision": alpha_precision,
            "beta_recall": beta_recall,
        }
    )

    curves_df.to_csv(
        filename,
        index=False,
    )

    return curves_df


def plot_curves(
    metrics: dict,
    title: str,
    filename: str,
):
    """
    Plot alpha-precision and beta-recall curves.
    """

    alphas = metrics["alphas"]

    alpha_precision = (
        metrics["alpha_precision_curve"]
    )

    beta_recall = (
        metrics["beta_recall_curve"]
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.plot(
        alphas,
        alpha_precision,
        label="Alpha-precision",
    )

    ax.plot(
        alphas,
        beta_recall,
        label="Beta-recall",
    )

    ax.plot(
        alphas,
        alphas,
        linestyle="--",
        label="Ideal",
    )

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Metric")
    ax.set_title(title)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=150,
    )

    plt.close(fig)


def _main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Dataset 2 against Dataset 1 "
            "using a SVDD embedding."
        )
    )

    parser.add_argument(
        "--svdd-model",
        type=str,
        required=True,
        help=(
            "SVDD checkpoint or directory. "
            "If a directory is provided, best.ckpt is used."
        ),
    )

    parser.add_argument(
        "--dataset1-dir",
        type=str,
        required=True,
        help=(
            "Reference/original dataset. "
            "The SVDD must have been trained on this dataset."
        ),
    )

    parser.add_argument(
        "--dataset2-dir",
        type=str,
        required=True,
        help=(
            "Synthetic/reconstructed dataset."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./result/svdd_metrics",
        help=(
            "Directory where evaluation results are saved."
        ),
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=30,
        help=(
            "Number of points used to evaluate "
            "alpha-precision and beta-recall curves."
        ),
    )

    builder = ml_db.IemanjaBuilder()

    builder.add_argparse_args(parser)

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Setup
    # --------------------------------------------------------------

    torch.set_float32_matmul_precision("medium")

    ml_utils.set_seed()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    device = ml_device.get_available_device()

    print()
    print("=" * 70)
    print("SVDD GENERATIVE METRICS")
    print("=" * 70)

    print(f"SVDD      : {args.svdd_model}")
    print(f"Dataset 1 : {args.dataset1_dir}")
    print(f"Dataset 2 : {args.dataset2_dir}")
    print(f"Device    : {device}")
    print(f"Output    : {args.output_dir}")
    print(f"n_steps   : {args.n_steps}")

    print("=" * 70)

    # --------------------------------------------------------------
    # Load model
    # --------------------------------------------------------------

    model = load_model(
        args.svdd_model
    )

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------------
    # Build datasets
    # --------------------------------------------------------------

    print()
    print("Building Dataset 1...")

    dm1 = build_datamodule(
        builder=builder,
        args=args,
        dataset_dir=args.dataset1_dir,
    )

    print()
    print("Building Dataset 2...")

    dm2 = build_datamodule(
        builder=builder,
        args=args,
        dataset_dir=args.dataset2_dir,
    )

    # --------------------------------------------------------------
    # Verify pairing
    # --------------------------------------------------------------

    verify_split_parity(
        dm1,
        dm2,
    )

    dataset1_name = "original"
    dataset2_name = "reconstructed"

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    metrics_rows = []

    for split in ["val", "test"]:

        print()
        print("=" * 70)
        print(f"Evaluating {split}")
        print("=" * 70)

        if split == "val":

            real_loader = dm1.val_dataloader(
                shuffle=False
            )

            synthetic_loader = dm2.val_dataloader(
                shuffle=False
            )

        else:

            real_loader = dm1.test_dataloader(
                shuffle=False
            )

            synthetic_loader = dm2.test_dataloader(
                shuffle=False
            )

        print(
            f"Reference : {dataset1_name}"
        )

        print(
            f"Synthetic : {dataset2_name}"
        )

        metrics = evaluate_split(
            model=model,
            real_dataloader=real_loader,
            synthetic_dataloader=synthetic_loader,
        )

        # ----------------------------------------------------------
        # Print metrics
        # ----------------------------------------------------------

        print()
        print(
            f"Alpha-precision : "
            f"{metrics['alpha_precision']:.6f}"
        )

        print(
            f"Beta-recall     : "
            f"{metrics['beta_recall']:.6f}"
        )

        print(
            f"Authenticity    : "
            f"{metrics['authenticity']:.6f}"
        )

        # ----------------------------------------------------------
        # Save scalar metrics
        # ----------------------------------------------------------

        metrics_rows.append(
            {
                "split": split,
                "dataset1": dataset1_name,
                "dataset2": dataset2_name,
                "alpha_precision":
                    metrics["alpha_precision"],
                "beta_recall":
                    metrics["beta_recall"],
                "authenticity":
                    metrics["authenticity"],
            }
        )

        # ----------------------------------------------------------
        # Save curves
        # ----------------------------------------------------------

        curves_file = os.path.join(
            args.output_dir,
            f"curves_{split}.csv",
        )

        save_curves(
            metrics=metrics,
            filename=curves_file,
            title=f"SVDD generative metrics - {split}",
        )

        # ----------------------------------------------------------
        # Plot curves
        # ----------------------------------------------------------

        plot_file = os.path.join(
            args.output_dir,
            f"curves_{split}.png",
        )

        plot_curves(
            metrics=metrics,
            title=(
                f"Alpha-precision / Beta-recall - "
                f"{split}"
            ),
            filename=plot_file,
        )

    # --------------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------------

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_file = os.path.join(
        args.output_dir,
        "metrics.csv",
    )

    metrics_df.to_csv(
        metrics_file,
        index=False,
    )

    # --------------------------------------------------------------
    # Final output
    # --------------------------------------------------------------

    print()
    print("=" * 70)
    print("METRICS")
    print("=" * 70)

    print(
        metrics_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)

    print(
        f"Metrics : "
        f"{metrics_file}"
    )

    print(
        f"Val curves : "
        f"{args.output_dir}/curves_val.csv"
    )

    print(
        f"Test curves: "
        f"{args.output_dir}/curves_test.csv"
    )

    print(
        f"Val plot   : "
        f"{args.output_dir}/curves_val.png"
    )

    print(
        f"Test plot  : "
        f"{args.output_dir}/curves_test.png"
    )

    print("=" * 70)


if __name__ == "__main__":
    _main()