#!/usr/bin/env python3
"""
Controlled benchmark for SVDD + alpha-precision / beta-recall /
authenticity.

The benchmark is completely independent of Iemanja.

A synthetic reference dataset is generated directly in the input
space. The SVDD is trained on the reference training set and the
controlled cases A-F are evaluated in the learned SVDD embedding.

Cases:

    A - Same distribution
        Independent samples from the same distribution.

    B - Same distribution + small noise
        Reference samples with small Gaussian perturbation.

    C - Mode collapse
        Samples generated from a single mode of the reference
        distribution.

    D - Shifted distribution
        Reference distribution shifted away from its original
        location.

    E - Near-training copies
        Training samples with very small perturbation.

    F - Excessively broad distribution
        Reference distribution with increased variance.

Default:
    Run all cases A-F.

Example:

    python benchmark_svdd.py \
        --input-dim 32 \
        --base-dir ./result/svdd_benchmark

Selected cases:

    python benchmark_svdd.py \
        --input-dim 32 \
        --base-dir ./result/svdd_benchmark \
        --cases A C E
"""

import os
import argparse
import shutil

import numpy as np
import pandas as pd
import sklearn.cluster as sk_cluster
import matplotlib.pyplot as plt

import torch
import torch.utils.data as torch_data

import lightning
import lightning.pytorch.loggers as lightning_log
import lightning.pytorch.callbacks as lightning_call

import lps_ml.model.svdd as ml_svdd
import lps_ml.utils.device as ml_device
import lps_ml.utils.general as ml_utils


# ============================================================================
# CASE DEFINITIONS
# ============================================================================

CASE_DESCRIPTIONS = {
    "A": {
        "name": "Same distribution",
        "reference": "Reference distribution",
        "generated": "Independent samples from the same distribution",
        "expected": "alpha-P ≈ 1, beta-R ≈ 1",
    },

    "B": {
        "name": "Same distribution + small noise",
        "reference": "Reference distribution",
        "generated": "Reference samples + small Gaussian noise",
        "expected": "alpha-P high, beta-R high",
    },

    "C": {
        "name": "Mode collapse",
        "reference": "Reference distribution",
        "generated": "Samples from a single mode",
        "expected": "alpha-P high, beta-R low",
    },

    "D": {
        "name": "Shifted distribution",
        "reference": "Reference distribution",
        "generated": "Reference distribution shifted",
        "expected": "alpha-P low, beta-R low",
    },

    "E": {
        "name": "Near-training copies",
        "reference": "Reference distribution",
        "generated": "Training samples + very small noise",
        "expected": "alpha-P may be high, authenticity low",
    },

    "F": {
        "name": "Excessively broad distribution",
        "reference": "Reference distribution",
        "generated": "Reference distribution with increased variance",
        "expected": "alpha-P tends to decrease, beta-R may remain high",
    },
}


# ============================================================================
# EARLY STOPPING
# ============================================================================

class WarmupEarlyStopping(
    lightning_call.EarlyStopping
):
    """Early stopping that ignores validation before SVDD warm-up."""

    def __init__(
        self,
        warmup_epochs: int,
        *args,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        self.warmup_epochs = warmup_epochs

    def on_validation_end(
        self,
        trainer,
        pl_module,
    ):
        """Run early stopping only after warm-up."""

        if trainer.current_epoch < self.warmup_epochs:
            return

        super().on_validation_end(
            trainer,
            pl_module,
        )


# ============================================================================
# ARGUMENTS
# ============================================================================

def _add_svdd_args(
    parser: argparse.ArgumentParser,
):
    """Add SVDD training arguments."""

    group = parser.add_argument_group(
        "SVDD",
        "SVDD architecture and training parameters",
    )

    group.add_argument(
        "--svdd-hidden-channels",
        type=int,
        nargs="+",
        default=[],
        help=(
            "Number of neurons in each hidden MLP layer. "
            "Example: --svdd-hidden-channels 128 64"
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
        choices=[
            e.name
            for e in ml_svdd.SVDDLoss
        ],
        help="SVDD objective function.",
    )

    group.add_argument(
        "--svdd-nu",
        type=float,
        default=0.1,
        help="Expected fraction of outliers.",
    )

    group.add_argument(
        "--svdd-lr",
        type=float,
        default=1e-3,
        help="SVDD learning rate.",
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
        help="Minimum absolute value for center initialization.",
    )

    group.add_argument(
        "--svdd-warmup-epochs",
        type=int,
        default=10,
        help="Number of warm-up epochs.",
    )


# ============================================================================
# CASE TABLE
# ============================================================================

def print_case_table():

    rows = []

    for case, description in CASE_DESCRIPTIONS.items():

        rows.append(
            {
                "Case": case,
                "Description": description["name"],
                "Dataset 1": description["reference"],
                "Dataset 2": description["generated"],
                "Expected": description["expected"],
            }
        )

    df = pd.DataFrame(rows)

    print()
    print("=" * 120)
    print("CONTROLLED SVDD BENCHMARK")
    print("=" * 120)
    print()

    print(
        df.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print()


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_reference_dataset(
    n_samples,
    input_dim,
    n_modes,
    mode_std,
    mode_distance,
    seed,
):
    """
    Generate a multimodal Gaussian reference distribution.

    The modes are arranged along the first input dimension.

    This gives a controlled distribution for the benchmark:

        p(x) = mixture of Gaussians
    """

    rng = np.random.default_rng(
        seed
    )

    samples_per_mode = (
        n_samples // n_modes
    )

    remainder = (
        n_samples % n_modes
    )

    samples = []

    for mode in range(n_modes):

        n = samples_per_mode

        if mode < remainder:
            n += 1

        center = np.zeros(
            input_dim,
            dtype=np.float32,
        )

        # Spread the modes along dimension 0.
        center[0] = (
            mode
            - (n_modes - 1) / 2
        ) * mode_distance

        x = (
            rng.normal(
                loc=center,
                scale=mode_std,
                size=(n, input_dim),
            )
            .astype(np.float32)
        )

        samples.append(x)

    x = np.concatenate(
        samples,
        axis=0,
    )

    rng.shuffle(x)

    return torch.from_numpy(
        x
    )


def split_reference_dataset(
    x,
    train_fraction,
    val_fraction,
):

    n = len(x)

    n_train = int(
        n * train_fraction
    )

    n_val = int(
        n * val_fraction
    )

    x_train = x[
        :n_train
    ]

    x_val = x[
        n_train:n_train + n_val
    ]

    x_test = x[
        n_train + n_val:
    ]

    return (
        x_train,
        x_val,
        x_test,
    )


# ============================================================================
# DATALOADER
# ============================================================================

def make_loader(
    x,
    batch_size,
):

    y = torch.zeros(
        len(x),
        dtype=torch.long,
    )

    dataset = torch_data.TensorDataset(
        x,
        y,
    )

    return torch_data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


# ============================================================================
# DATA / EMBEDDING EXTRACTION
# ============================================================================

def extract_embedding(
    model,
    x,
    device,
    batch_size,
):

    loader = make_loader(
        x=x,
        batch_size=batch_size,
    )

    model.eval()

    embeddings = []

    with torch.no_grad():

        for batch in loader:

            batch_x = batch[0].to(
                device
            )

            z = model(
                batch_x
            )

            embeddings.append(
                z.cpu()
            )

    return torch.cat(
        embeddings,
        dim=0,
    )


def extract_distances(
    model,
    x,
    device,
    batch_size,
):

    loader = make_loader(
        x=x,
        batch_size=batch_size,
    )

    model.eval()

    distances = []

    with torch.no_grad():

        for batch in loader:

            batch_x = batch[0].to(
                device
            )

            z = model(
                batch_x
            )

            squared_distance = (
                model.squared_distance(z)
            )

            distance = torch.sqrt(
                torch.clamp(
                    squared_distance,
                    min=0.0,
                )
            )

            distances.append(
                distance.cpu().numpy()
            )

    return np.concatenate(
        distances
    )


# ============================================================================
# DISTANCE STATISTICS
# ============================================================================

def build_distance_statistics(
    distances,
):

    factors = [
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
    ]

    statistics = {
        "mean_distance":
            float(np.mean(distances)),

        "std_distance":
            float(np.std(distances)),
    }

    for factor in factors:

        statistics[
            f"q{factor:02d}"
        ] = float(
            np.quantile(
                distances,
                factor / 100.0,
            )
        )

    return statistics


def plot_distance_histogram(
    distances,
    radius,
    dataset,
    filename,
):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

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
        label=(
            f"SVDD radius = "
            f"{radius:.4f}"
        ),
    )

    ax.set_xlabel(
        "Distance to SVDD center"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        f"SVDD distance distribution - "
        f"{dataset}"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
    )

    plt.close(fig)


# ============================================================================
# CONTROLLED CASES
# ============================================================================

def build_case_A(
    n_samples,
    input_dim,
    n_modes,
    mode_std,
    mode_distance,
    seed,
    **kwargs,
):
    """
    Same distribution.

    Generate independent samples from the same underlying
    reference distribution.
    """

    return generate_reference_dataset(
        n_samples=n_samples,
        input_dim=input_dim,
        n_modes=n_modes,
        mode_std=mode_std,
        mode_distance=mode_distance,
        seed=seed + 1000,
    )

def build_case_B(
    x_val,
    noise_std,
    seed,
    **kwargs,
):

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    noise = (
        torch.randn(
            x_val.shape,
            generator=generator,
        )
        * noise_std
    )

    return (
        x_val
        + noise
    )

def build_case_C(
    input_dim,
    mode_std,
    mode_distance,
    n_modes,
    n_samples,
    seed,
    **kwargs,
):
    """
    Mode collapse.

    Generate samples exclusively from one of the known
    Gaussian modes of the reference distribution.
    """

    rng = np.random.default_rng(seed)

    # Escolhe um modo de forma determinística
    selected_mode = n_modes // 2

    rng = np.random.default_rng(seed)

    global_center = rng.uniform(
        low=-20.0,
        high=20.0,
        size=input_dim,
    ).astype(np.float32)

    center = global_center.copy()

    center[0] = (
        selected_mode
        - (n_modes - 1) / 2
    ) * mode_distance

    x = rng.normal(
        loc=center,
        scale=mode_std,
        size=(n_samples, input_dim),
    ).astype(np.float32)

    print(f"Selected reference mode: {selected_mode}")

    return torch.from_numpy(x)

def build_case_D(
    x_val,
    shift_factor,
    **kwargs,
):
    """
    Shift the entire validation distribution.
    """

    mean = torch.mean(
        x_val,
        dim=0,
        keepdim=True,
    )

    std = torch.std(
        x_val,
        dim=0,
        keepdim=True,
    )

    shift = torch.zeros_like(mean)
    shift[:, 0] = shift_factor * std[:, 0]

    return x_val + shift


def build_case_E(
    x_train,
    noise_std,
    n_samples,
    seed,
    **kwargs,
):
    """
    Near-training copies.

    These samples are derived directly from the training
    observations and therefore should stress authenticity.
    """

    if n_samples <= 0:
        n_samples = len(x_train)

    n_samples = min(
        n_samples,
        len(x_train),
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    indices = torch.randperm(
        len(x_train),
        generator=generator,
    )[:n_samples]

    x = x_train[
        indices
    ].clone()

    noise = (
        torch.randn(
            x.shape,
            generator=generator,
        )
        * noise_std
    )

    return (
        x
        + noise
    )


def build_case_F(
    x_val,
    scale_factor,
    **kwargs,
):
    """
    Increase the variance of the reference distribution.
    """

    mean = torch.mean(
        x_val,
        dim=0,
        keepdim=True,
    )

    return (
        mean
        + scale_factor
        * (x_val - mean)
    )


def build_case(
    case,
    **kwargs,
):

    builders = {
        "A": build_case_A,
        "B": build_case_B,
        "C": build_case_C,
        "D": build_case_D,
        "E": build_case_E,
        "F": build_case_F,
    }

    return builders[case](
        **kwargs
    )

# ============================================================================
# CURVES
# ============================================================================

def save_curves(
    metrics,
    filename,
):

    curves_df = pd.DataFrame(
        {
            "alpha":
                metrics["alphas"],

            "alpha_precision":
                metrics[
                    "alpha_precision_curve"
                ],

            "beta_recall":
                metrics[
                    "beta_recall_curve"
                ],
        }
    )

    curves_df.to_csv(
        filename,
        index=False,
    )


def plot_curves(
    metrics,
    case,
    filename,
):

    alphas = metrics[
        "alphas"
    ]

    alpha_precision = metrics[
        "alpha_precision_curve"
    ]

    beta_recall = metrics[
        "beta_recall_curve"
    ]

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

    ax.set_xlabel(
        "Alpha"
    )

    ax.set_ylabel(
        "Metric"
    )

    ax.set_title(
        f"Case {case}: "
        f"{CASE_DESCRIPTIONS[case]['name']}"
    )

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.grid(True)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=150,
    )

    plt.close(fig)

def plot_latent_space_2d(
    model,
    x_reference,
    x_generated,
    device,
    batch_size,
    case,
    filename,
):
    """
    Plot reference and generated samples in the 2-D SVDD latent space.

    This visualization is only meaningful when latent_dim == 2.
    """

    z_reference = extract_embedding(
        model=model,
        x=x_reference,
        device=device,
        batch_size=batch_size,
    ).numpy()

    z_generated = extract_embedding(
        model=model,
        x=x_generated,
        device=device,
        batch_size=batch_size,
    ).numpy()

    center = (
        model.center
        .detach()
        .cpu()
        .numpy()
    )

    radius = model.radius.item()

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    # --------------------------------------------------------------
    # Reference
    # --------------------------------------------------------------

    ax.scatter(
        z_reference[:, 0],
        z_reference[:, 1],
        s=10,
        alpha=0.35,
        label="Reference",
    )

    # --------------------------------------------------------------
    # Generated
    # --------------------------------------------------------------

    ax.scatter(
        z_generated[:, 0],
        z_generated[:, 1],
        s=12,
        alpha=0.65,
        marker="x",
        label="Generated",
    )

    # --------------------------------------------------------------
    # SVDD center
    # --------------------------------------------------------------

    ax.scatter(
        center[0],
        center[1],
        s=120,
        marker="*",
        label="SVDD center",
    )

    # --------------------------------------------------------------
    # SVDD hypersphere
    # --------------------------------------------------------------

    circle = plt.Circle(
        (
            center[0],
            center[1],
        ),
        radius,
        fill=False,
        linestyle="--",
        linewidth=2,
        label="SVDD radius",
    )

    ax.add_patch(circle)

    # --------------------------------------------------------------
    # Labels
    # --------------------------------------------------------------

    description = CASE_DESCRIPTIONS[case]

    ax.set_title(
        f"Case {case}: "
        f"{description['name']}\n"
        f"SVDD latent space"
    )

    ax.set_xlabel(
        "SVDD latent dimension 1"
    )

    ax.set_ylabel(
        "SVDD latent dimension 2"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    ax.set_aspect(
        "equal",
        adjustable="datalim",
    )

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
    )

    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def _main():

    parser = argparse.ArgumentParser(
        description=(
            "Controlled benchmark for validating "
            "SVDD and alpha-precision, beta-recall "
            "and authenticity."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Input space
    # ------------------------------------------------------------------

    parser.add_argument(
        "--input-dim",
        type=int,
        required=True,
        help=(
            "Dimension of the input vector."
        ),
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="./result/svdd_benchmark",
        help=(
            "Base output directory.\n"
            "Outputs:\n"
            "  base_dir/log\n"
            "  base_dir/train\n"
            "  base_dir/metrics"
        ),
    )

    # ------------------------------------------------------------------
    # Cases
    # ------------------------------------------------------------------

    parser.add_argument(
        "--cases",
        type=str,
        nargs="+",
        choices=list(
            CASE_DESCRIPTIONS.keys()
        ),
        default=list(
            CASE_DESCRIPTIONS.keys()
        ),
        help=(
            "Cases to execute.\n\n"

            "A - Same distribution\n"
            "    Independent samples from the same distribution.\n\n"

            "B - Same distribution + small noise\n"
            "    Same distribution with small Gaussian noise.\n\n"

            "C - Mode collapse\n"
            "    Samples from a single mode.\n\n"

            "D - Shifted distribution\n"
            "    Distribution shifted away from reference.\n\n"

            "E - Near-training copies\n"
            "    Training samples + very small noise.\n\n"

            "F - Excessively broad distribution\n"
            "    Distribution with increased variance.\n\n"

            "Default: A B C D E F\n\n"

            "Example:\n"
            "    --cases A C E"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size.",
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=30,
        help=(
            "Number of points used for "
            "alpha-precision/beta-recall curves."
        ),
    )

    # ------------------------------------------------------------------
    # Reference distribution
    # ------------------------------------------------------------------

    group = parser.add_argument_group(
        "Reference distribution",
        "Parameters of the synthetic reference distribution.",
    )

    group.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Total number of reference samples.",
    )

    group.add_argument(
        "--n-modes",
        type=int,
        default=4,
        help="Number of Gaussian modes.",
    )

    group.add_argument(
        "--mode-std",
        type=float,
        default=1.0,
        help="Standard deviation of each Gaussian mode.",
    )

    group.add_argument(
        "--mode-distance",
        type=float,
        default=6.0,
        help=(
            "Distance between consecutive Gaussian modes "
            "along the first dimension."
        ),
    )

    group.add_argument(
        "--train-fraction",
        type=float,
        default=0.6,
        help="Fraction of samples used for SVDD training.",
    )

    group.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of samples used for validation.",
    )

    # ------------------------------------------------------------------
    # Controlled cases
    # ------------------------------------------------------------------

    group = parser.add_argument_group(
        "Controlled cases",
        "Parameters controlling cases B-F.",
    )

    group.add_argument(
        "--case-noise-std",
        type=float,
        default=0.01,
        help=(
            "Gaussian noise standard deviation "
            "for cases B and E."
        ),
    )

    group.add_argument(
        "--case-shift-factor",
        type=float,
        default=2.0,
        help=(
            "Shift magnitude in standard deviations "
            "for case D."
        ),
    )

    group.add_argument(
        "--case-scale-factor",
        type=float,
        default=2.0,
        help=(
            "Variance expansion factor "
            "for case F."
        ),
    )

    group.add_argument(
        "--case-n-clusters",
        type=int,
        default=4,
        help=(
            "Number of KMeans clusters used "
            "for case C."
        ),
    )

    group.add_argument(
        "--case-n-samples",
        type=int,
        default=2000,
        help=(
            "Number of training samples used "
            "for case E. "
            "0 means all training samples."
        ),
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=2000,
        help="Maximum number of SVDD epochs.",
    )

    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
        help="Minimum validation loss improvement.",
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=200,
        help="Early stopping patience.",
    )

    _add_svdd_args(
        parser
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    torch.set_float32_matmul_precision(
        "medium"
    )

    ml_utils.set_seed()

    print_case_table()

    print(
        f"Input dimension : {args.input_dim}"
    )

    print(
        f"Selected cases  : "
        f"{', '.join(args.cases)}"
    )

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------

    log_dir = os.path.join(
        args.base_dir,
        "log",
    )

    train_dir = os.path.join(
        args.base_dir,
        "train",
    )

    metrics_dir = os.path.join(
        args.base_dir,
        "metrics",
    )

    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    os.makedirs(
        train_dir,
        exist_ok=True,
    )

    os.makedirs(
        metrics_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Generate reference dataset
    # ------------------------------------------------------------------

    print()
    print(
        ml_utils.format_header(
            70,
            "REFERENCE DATASET",
        )
    )

    x = generate_reference_dataset(
        n_samples=args.n_samples,
        input_dim=args.input_dim,
        n_modes=args.n_modes,
        mode_std=args.mode_std,
        mode_distance=args.mode_distance,
        seed=args.seed,
    )

    x_train, x_val, x_test = (
        split_reference_dataset(
            x=x,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
    )

    print(
        f"Total  : {len(x)}"
    )

    print(
        f"Train  : {len(x_train)}"
    )

    print(
        f"Val    : {len(x_val)}"
    )

    print(
        f"Test   : {len(x_test)}"
    )

    print(
        f"Shape  : {tuple(x.shape)}"
    )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    train_loader = make_loader(
        x_train,
        args.batch_size,
    )

    val_loader = make_loader(
        x_val,
        args.batch_size,
    )

    # ------------------------------------------------------------------
    # SVDD
    # ------------------------------------------------------------------

    print()
    print(
        ml_utils.format_header(
            70,
            "SVDD TRAINING",
        )
    )

    model = ml_svdd.SVDDMLP(
        input_shape=(
            args.input_dim,
        ),

        hidden_channels=
            args.svdd_hidden_channels,

        latent_dim=
            args.svdd_latent_dim,

        loss=
            ml_svdd.SVDDLoss[
                args.svdd_loss
            ],

        nu=
            args.svdd_nu,

        lr=
            args.svdd_lr,

        weight_decay=
            args.svdd_weight_decay,

        center_eps=
            args.svdd_center_eps,

        warm_up_n_epochs=
            args.svdd_warmup_epochs,
    )

    checkpoint_cb = (
        lightning_call.ModelCheckpoint(
            dirpath=train_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best",
        )
    )

    callbacks = [

        checkpoint_cb,

        WarmupEarlyStopping(
            warmup_epochs=
                args.svdd_warmup_epochs,

            monitor="val/loss",

            min_delta=
                args.early_stopping_min_delta,

            patience=
                args.early_stopping_patience,

            verbose=True,

            mode="min",
        ),

        lightning_call.LearningRateMonitor(
            logging_interval="epoch"
        ),
    ]

    logger = (
        lightning_log.TensorBoardLogger(
            save_dir=log_dir,
            name="svdd_benchmark",
        )
    )

    trainer = lightning.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
    )

    print("x shape:", x.shape)
    print("x_train shape:", x_train.shape)
    print("x_val shape:", x_val.shape)
    print("x_test shape:", x_test.shape)

    print("model:", model.embedder)

    model.initialize_center(train_loader)

    with torch.no_grad():

        z_train = extract_embedding(
            model=model,
            x=x_train,
            device=model.device,
            batch_size=args.batch_size,
        )

    print()
    print("INITIAL LATENT")
    print("mean:", z_train.mean(dim=0).numpy())
    print("std :", z_train.std(dim=0).numpy())
    print("min :", z_train.min(dim=0).values.numpy())
    print("max :", z_train.max(dim=0).values.numpy())

    print()
    print("INITIAL CENTER")
    print(model.center.detach().cpu().numpy())

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    z_train = extract_embedding(
        model=model,
        x=x_train,
        device=model.device,
        batch_size=args.batch_size,
    )

    print()
    print("FINAL LATENT")
    print("mean:", z_train.mean(dim=0).numpy())
    print("std :", z_train.std(dim=0).numpy())
    print("min :", z_train.min(dim=0).values.numpy())
    print("max :", z_train.max(dim=0).values.numpy())

    print()
    print("FINAL CENTER")
    print(model.center.detach().cpu().numpy())

    print()
    print("RADIUS")
    print(model.radius.item())

    # ------------------------------------------------------------------
    # Save checkpoints
    # ------------------------------------------------------------------

    model = (
        ml_svdd.SVDDMLP.load_from_checkpoint(
            os.path.join(
                train_dir,
                "best.ckpt",
            )
        )
    )

    device = (
        ml_device.get_available_device()
    )

    model = model.to(
        device
    )

    model.eval()

    # ------------------------------------------------------------------
    # Distance analysis
    # ------------------------------------------------------------------

    print()
    print(
        ml_utils.format_header(
            70,
            "SVDD DISTANCES",
        )
    )

    split_data = {
        "train": x_train,
        "val": x_val,
        "test": x_test,
    }

    statistics_rows = []

    for split, x_split in (
        split_data.items()
    ):

        distances = extract_distances(
            model=model,
            x=x_split,
            device=device,
            batch_size=args.batch_size,
        )

        pd.DataFrame(
            {
                "distance": distances
            }
        ).to_csv(
            os.path.join(
                train_dir,
                f"distances_{split}.csv",
            ),
            index=False,
        )

        statistics = (
            build_distance_statistics(
                distances
            )
        )

        statistics[
            "split"
        ] = split

        statistics_rows.append(
            statistics
        )

        plot_distance_histogram(
            distances=distances,
            radius=model.radius.item(),
            dataset=split,
            filename=os.path.join(
                train_dir,
                f"distance_histogram_{split}.png",
            ),
        )

    statistics_df = pd.DataFrame(
        statistics_rows
    )

    statistics_df.to_csv(
        os.path.join(
            train_dir,
            "distance_statistics.csv",
        ),
        index=False,
    )

    # ------------------------------------------------------------------
    # Evaluate cases
    # ------------------------------------------------------------------

    summary_rows = []

    for case in args.cases:

        print()
        print("=" * 70)

        print(
            f"CASE {case}: "
            f"{CASE_DESCRIPTIONS[case]['name']}"
        )

        print("=" * 70)

        x_case = build_case(
            case=case,
            model=model,
            x_train=x_train,
            x_val=x_val,
            device=device,
            n_samples=args.case_n_samples,
            input_dim=args.input_dim,
            n_modes=args.n_modes,
            mode_std=args.mode_std,
            mode_distance=args.mode_distance,
            noise_std=args.case_noise_std,
            shift_factor=args.case_shift_factor,
            scale_factor=args.case_scale_factor,
            n_clusters=args.case_n_clusters,
            seed=args.seed,
        )

        print(
            f"Reference samples : "
            f"{len(x_val)}"
        )

        print(
            f"Generated samples : "
            f"{len(x_case)}"
        )

        reference_loader = make_loader(
            x_val,
            args.batch_size,
        )

        synthetic_loader = make_loader(
            x_case,
            args.batch_size,
        )

        metrics = (
            model.calculate_alpha_beta_authenticity(
                real_dataloader=
                    reference_loader,

                synthetic_dataloader=
                    synthetic_loader,

                n_steps=
                    args.n_steps,
            )
        )

        alpha_precision = (
            metrics["alpha_precision"]
        )

        beta_recall = (
            metrics["beta_recall"]
        )

        authenticity = (
            metrics["authenticity"]
        )

        print()
        print(
            f"Alpha-precision : "
            f"{alpha_precision:.6f}"
        )

        print(
            f"Beta-recall     : "
            f"{beta_recall:.6f}"
        )

        print(
            f"Authenticity    : "
            f"{authenticity:.6f}"
        )

        # --------------------------------------------------------------
        # Case CSV
        # --------------------------------------------------------------

        description = (
            CASE_DESCRIPTIONS[case]
        )

        case_df = pd.DataFrame(
            [
                {
                    "case": case,

                    "description":
                        description["name"],

                    "dataset1":
                        description["reference"],

                    "dataset2":
                        description["generated"],

                    "expected":
                        description["expected"],

                    "alpha_precision":
                        alpha_precision,

                    "beta_recall":
                        beta_recall,

                    "authenticity":
                        authenticity,

                    "n_reference":
                        len(x_val),

                    "n_generated":
                        len(x_case),
                }
            ]
        )

        case_df.to_csv(
            os.path.join(
                metrics_dir,
                f"case_{case}.csv",
            ),
            index=False,
        )

        # --------------------------------------------------------------
        # Curves
        # --------------------------------------------------------------

        save_curves(
            metrics=metrics,
            filename=os.path.join(
                metrics_dir,
                f"curves_{case}.csv",
            ),
        )

        plot_curves(
            metrics=metrics,
            case=case,
            filename=os.path.join(
                metrics_dir,
                f"curves_{case}.png",
            ),
        )

        summary_rows.append(
            {
                "case": case,

                "description":
                    description["name"],

                "expected":
                    description["expected"],

                "alpha_precision":
                    alpha_precision,

                "beta_recall":
                    beta_recall,

                "authenticity":
                    authenticity,

                "n_reference":
                    len(x_val),

                "n_generated":
                    len(x_case),
            }
        )

        if args.svdd_latent_dim == 2:

            latent_plot_file = os.path.join(
                metrics_dir,
                f"latent_{case}.png",
            )

            plot_latent_space_2d(
                model=model,
                x_reference=x_val,
                x_generated=x_case,
                device=device,
                batch_size=args.batch_size,
                case=case,
                filename=latent_plot_file,
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = os.path.join(
        metrics_dir,
        "summary.csv",
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    print()
    print(
        ml_utils.format_header(
            120,
            "BENCHMARK RESULTS",
        )
    )

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        ml_utils.format_header(
            70,
            "OUTPUT",
        )
    )

    print(
        f"Base directory : "
        f"{args.base_dir}"
    )

    print(
        f"Log            : "
        f"{log_dir}"
    )

    print(
        f"Training       : "
        f"{train_dir}"
    )

    print(
        f"Metrics        : "
        f"{metrics_dir}"
    )

    print(
        f"Summary        : "
        f"{summary_file}"
    )


if __name__ == "__main__":
    _main()
