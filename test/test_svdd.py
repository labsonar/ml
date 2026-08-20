"""
SVDD training script for Iemanja dataset.
"""

import os
import argparse
import shutil
import typing

import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt

import torch
import torch.utils.data as torch_data

import lightning
import lightning.pytorch.loggers as lightning_log
import lightning.pytorch.callbacks as lightning_call

import lps_ml.core.cv as ml_cv
import lps_ml.datasets as ml_db
import lps_ml.utils.device as ml_device
import lps_ml.utils.general as ml_utils
import lps_ml.model.svdd as ml_svdd


def _add_svdd_args(parser: argparse.ArgumentParser):
    """Add SVDD-MLP command-line arguments."""

    group = parser.add_argument_group("SVDD", "Deep SVDD MLP architecture and training parameters")

    group.add_argument(
        "--svdd-hidden-channels",
        type=int,
        nargs="+",
        default=[],
        help=(
            "Number of neurons in each hidden MLP layer. "
            "Example: --svdd-hidden-channels []"
        ),
    )

    group.add_argument(
        "--svdd-latent-dim",
        type=int,
        default=32,
        help="Dimension of the SVDD embedding space.",
    )

    group.add_argument(
        "--svdd-loss",
        type=str,
        default=ml_svdd.SVDDLoss.SOFT_BOUNDARY.name,
        choices=[e.name for e in ml_svdd.SVDDLoss],
        help="SVDD objective function.",
    )

    group.add_argument(
        "--svdd-nu",
        type=float,
        default=0.1,
        help=(
            "Expected fraction of outliers. "
            "Must be in the interval (0, 1]."
        ),
    )

    group.add_argument(
        "--svdd-lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    group.add_argument(
        "--svdd-weight-decay",
        type=float,
        default=1e-6,
        help="L2 weight decay.",
    )

    group.add_argument(
        "--svdd-center-eps",
        type=float,
        default=1e-4,
        help="Minimum absolute value used for center initialization.",
    )

    group.add_argument(
        "--svdd-warmup-epochs",
        type=int,
        default=10,
        help=(
            "Number of initial epochs during which the radius "
            "is not updated."
        ),
    )

def _extract_distances(
    model: ml_svdd.SVDDMLP,
    dataloader: torch_data.DataLoader,
) -> np.ndarray:
    """
    Extract Euclidean distances from samples to the SVDD center.
    """

    device = ml_device.get_available_device()

    model = model.to(device)
    model.eval()

    distances = []

    with torch.no_grad():

        for batch in dataloader:

            if not isinstance(batch, (tuple, list)):
                raise RuntimeError(
                    "Iemanja dataloader should return (x, y) "
                    "for evaluation."
                )

            x = batch[0].to(device)

            z = model(x)

            squared_distance = model.squared_distance(z)

            distance = torch.sqrt(torch.clamp(squared_distance, min=0.0))
            distances.append(distance.cpu().numpy())

    if not distances:
        raise RuntimeError("Dataloader produced no samples.")

    return np.concatenate(distances)

def _build_distance_statistics(distances: np.ndarray) -> dict:
    """
    Build descriptive statistics for the SVDD distances.
    """

    factors = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    statistics = {
        "mean distance": float(np.mean(distances)),
        "std distance": float(np.std(distances)),
    }

    for factor in factors:
        statistics[f"q{factor:02d}"] = float(np.quantile(distances, factor/100.0))

    return statistics

def _evaluate_split(model: ml_svdd.SVDDMLP, dataloader: torch_data.DataLoader) -> pd.DataFrame:
    """Extract SVDD distances for one dataset split."""

    device = ml_device.get_available_device()

    model = model.to(device)
    model.eval()

    distances = []

    with torch.no_grad():

        for batch in dataloader:

            if not isinstance(batch, (tuple, list)):
                raise RuntimeError(
                    "Iemanja dataloader should return (x, y) "
                    "for evaluation."
                )

            x = ml_svdd.SVDDMLP._get_input(batch)
            x = x.to(device)

            z = model(x)

            distance = torch.sqrt(torch.clamp(model.squared_distance(z), min=0.0))
            distances.append(distance.cpu().numpy())

    return pd.DataFrame({"distance": np.concatenate(distances)})

def _plot_distance_histogram(distances: np.ndarray, radius: float, dataset: str, filename: str,):
    """
    Plot the SVDD distance distribution for one dataset split.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        distances,
        bins=50,
        density=True,
        alpha=0.7,
    )

    ax.axvline(
        radius,
        linestyle="--",
        linewidth=2,
        label=f"SVDD radius = {radius:.4f}",
    )

    ax.set_xlabel("Distance to SVDD center")
    ax.set_ylabel("Density")
    ax.set_title(f"SVDD distance distribution - {dataset}")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def _main():
    """Train SVDD-MLP on Iemanja."""

    builder = ml_db.IemanjaBuilder()

    parser = argparse.ArgumentParser(description="Train a Deep SVDD MLP on the Iemanja dataset.")

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=2000,
        help="Maximum number of training epochs.",
    )

    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
        help=(
            "Minimum validation loss improvement "
            "required to reset early stopping."
        ),
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=200,
        help=(
            "Number of validation epochs without improvement "
            "before stopping."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./result/svdd",
        help="Output directory.",
    )

    builder.add_argparse_args(parser=parser)
    _add_svdd_args(parser=parser)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    ml_utils.set_seed()

    dm = builder.from_argparse_args(args)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_last = os.path.join(output_dir, "last.ckpt")
    model_best = os.path.join(output_dir, "best.ckpt")

    log_dir = os.path.join(output_dir, "log")

    sample_shape = dm.get_sample_shape()

    print(ml_utils.format_header(60, "SVDD"))
    print(f"sample shape : {sample_shape}")

    model = ml_svdd.SVDDMLP(
        input_shape=sample_shape,
        hidden_channels=args.svdd_hidden_channels,
        latent_dim=args.svdd_latent_dim,
        loss=ml_svdd.SVDDLoss[args.svdd_loss],
        nu=args.svdd_nu,
        lr=args.svdd_lr,
        weight_decay=args.svdd_weight_decay,
        center_eps=args.svdd_center_eps,
        warm_up_n_epochs=args.svdd_warmup_epochs,
    )

    checkpoint_cb = lightning_call.ModelCheckpoint(
        dirpath=log_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
    )

    callbacks = [
        checkpoint_cb,

        lightning_call.EarlyStopping(
            monitor="val/loss",
            min_delta=args.early_stopping_min_delta,
            patience=args.early_stopping_patience,
            verbose=True,
            mode="min",
        ),

        lightning_call.LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = lightning_log.TensorBoardLogger(log_dir, name="iemanja_svdd")

    trainer = lightning.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, dm)

    shutil.copy2(checkpoint_cb.last_model_path, model_last)
    shutil.copy2(checkpoint_cb.best_model_path, model_best)

    # last_model = ml_svdd.SVDDMLP.load_from_checkpoint(model_last)
    model.eval()

    train_df = _evaluate_split(model, dm.train_dataloader())
    val_df = _evaluate_split(model, dm.val_dataloader())
    test_df = _evaluate_split(model, dm.test_dataloader())

    train_df.to_csv(os.path.join(output_dir, "distances_train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "distances_val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "distances_test.csv"), index=False)

    statistics = {
        "train": _build_distance_statistics(train_df["distance"].to_numpy()),
        "val": _build_distance_statistics(val_df["distance"].to_numpy()),
        "test": _build_distance_statistics(test_df["distance"].to_numpy()),
    }

    statistics_df = pd.DataFrame(statistics)
    statistics_df.to_csv(os.path.join(output_dir, "distance_statistics.csv"))

    print(ml_utils.format_header(60, "Results"))

    print(f"latent dimension : {args.svdd_latent_dim}")
    print(f"center norm      : {model.center.norm().item():.6f}")
    print(f"radius           : {model.radius.item():.6f}")

    print()
    print(ml_utils.format_header(60, "Distance statistics"))
    print(statistics_df)


    radius = model.radius.item()

    _plot_distance_histogram(
        distances=train_df["distance"].to_numpy(),
        radius=radius,
        dataset="train",
        filename=os.path.join(output_dir, "distance_histogram_train.png"),
    )

    _plot_distance_histogram(
        distances=val_df["distance"].to_numpy(),
        radius=radius,
        dataset="val",
        filename=os.path.join(output_dir, "distance_histogram_val.png"),
    )

    _plot_distance_histogram(
        distances=test_df["distance"].to_numpy(),
        radius=radius,
        dataset="test",
        filename=os.path.join(output_dir, "distance_histogram_test.png"),
    )

if __name__ == "__main__":
    _main()
