"""
Evaluate latent-space classification of VAE checkpoints over training.
"""

import os
import argparse
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score

import lps_ml.datasets as ml_db
import lps_ml.utils.general as ml_utils
import lps_ml.core.cv as ml_cv


def _extract_latent_and_labels(loader, latent_mode="flatten"):
    """
    Extract latent representations and labels from a DataLoader.
    """

    all_data = []
    all_labels = []

    first_batch = True

    for x, y in loader:

        if first_batch:
            print(f"\tlatent batch shape: {x.shape}")
            first_batch = False

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if x.ndim > 2:

            if latent_mode == "flatten":

                x = x.reshape(x.shape[0], -1)

            elif latent_mode == "samples":

                B, D, T = x.shape

                x = np.transpose(x, (0, 2, 1))
                x = x.reshape(B * T, D)

                y = np.repeat(y, T)

            else:
                raise ValueError(
                    f"Unknown latent_mode: {latent_mode}"
                )

        all_data.append(x)
        all_labels.append(y)

    if not all_data:
        raise RuntimeError("DataLoader produced no samples.")

    data = np.vstack(all_data)
    labels = np.concatenate(all_labels)

    return data, labels

def _get_loader(dm, fold_role):
    """
    Return the requested DataLoader from an Iemanja DataModule.
    """

    if fold_role == "TRAIN":
        return dm.train_dataloader()

    if fold_role == "VALIDATION":
        return dm.val_dataloader()

    if fold_role == "TEST":
        return dm.test_dataloader()

    raise ValueError(
        f"Unsupported fold role: {fold_role}"
    )

def evaluate_model(
    model_path,
    builder,
    args,
    latent_compactness=1024,
    latent_mode="flatten",
    max_iter=1000,
):
    """
    Evaluate one VAE checkpoint.
    """

    print(f"\nLoading VAE: {model_path}")

    # ---------------------------------------------------------------
    # Build Iemanja DataModule using this VAE
    # ---------------------------------------------------------------

    latent_dm = builder.from_argparse_args(
        args,
        model_path,
        latent_compactness,
    )

    # Keep this evaluation lightweight.
    latent_dm.batch_size = 1
    latent_dm.num_workers = 0

    latent_dm.setup()

    train_loader = latent_dm.train_dataloader()
    val_loader = latent_dm.val_dataloader()

    # ---------------------------------------------------------------
    # Extract training latent representations
    # ---------------------------------------------------------------

    print("Extracting training latent representations...")

    X_train, y_train = _extract_latent_and_labels(
        train_loader,
        latent_mode=latent_mode,
    )

    print(
        f"\tX_train: {X_train.shape}"
        f"\n\ty_train: {y_train.shape}"
        f"\n\tclasses: {np.unique(y_train)}"
    )

    # ---------------------------------------------------------------
    # Extract validation latent representations
    # ---------------------------------------------------------------

    print("Extracting validation latent representations...")

    X_val, y_val = _extract_latent_and_labels(
        val_loader,
        latent_mode=latent_mode,
    )

    print(
        f"\tX_val:   {X_val.shape}"
        f"\n\ty_val:   {y_val.shape}"
    )

    # ---------------------------------------------------------------
    # Train simple linear classifier
    # ---------------------------------------------------------------

    print("Training LogisticRegression...")

    classifier = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
    )

    classifier.fit(X_train, y_train)

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    print("Evaluating validation set...")

    y_pred = classifier.predict(X_val)

    balanced_accuracy = balanced_accuracy_score(
        y_val,
        y_pred,
    )

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro",
    )

    print(
        f"\tBalanced Accuracy: {balanced_accuracy:.6f}"
        f"\n\tMacro F1:           {macro_f1:.6f}"
    )

    return {
        "balanced_accuracy": float(balanced_accuracy),
        "macro_f1": float(macro_f1),
    }


def plot_metric(
    results,
    model_ids,
    steps,
    metric,
    output_path,
):
    """
    Plot one validation metric as a function of training step.
    """

    plt.figure(figsize=(9, 6))

    for model_id in model_ids:

        values = [
            results
            .get(model_id, {})
            .get(step, {})
            .get(metric, np.nan)
            for step in steps
        ]

        plt.plot(
            steps,
            values,
            marker="o",
            label=model_id,
        )

    plt.xlabel("Training step")

    if metric == "balanced_accuracy":
        ylabel = "Balanced Accuracy"
        title = "Latent-space classification — Balanced Accuracy"

    elif metric == "macro_f1":
        ylabel = "Macro F1"
        title = "Latent-space classification — Macro F1"

    else:
        ylabel = metric
        title = metric

    plt.ylabel(ylabel)
    plt.title(title)

    plt.ylim(0.0, 1.0)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()


def _main():

    # -----------------------------------------------------------------
    # Dataset builder
    #
    # vae_exclusive=True makes the DataModule operate on the VAE
    # latent representation rather than the original waveform.
    # -----------------------------------------------------------------

    builder = ml_db.IemanjaBuilder(
        vae_exclusive=True,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VAE latent-space classification over training "
            "steps using LogisticRegression."
        )
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help=(
            "Directory containing VAE TorchScript checkpoints "
            "named {model_id}_ep{step}.ts"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./result/vae_latent_classification",
        help="Directory to save CSV and plots.",
    )

    parser.add_argument(
        "--latent-compactness",
        type=int,
        default=1024,
        help="VAE latent compactness.",
    )

    parser.add_argument(
        "--latent-mode",
        type=str,
        default="flatten",
        choices=["flatten", "samples"],
        help="How latent tensors are converted into classifier samples.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of LogisticRegression iterations.",
    )

    builder.add_argparse_args(
        parser=parser,
    )

    args = parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    ml_utils.set_seed()

    # -----------------------------------------------------------------
    # Models and training steps
    # -----------------------------------------------------------------

    # model_ids = ["ch1", "ch5", "ch10", "ch50", "ch100", "ch5_s1", "ch10_s1", "ch10_s5", "ch100_s50", "default"]
    model_ids = ["ch5", "ch5_s1", "default"]
    steps = [200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    # -----------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------

    results = collections.defaultdict(dict)

    rows = []

    # -----------------------------------------------------------------
    # Evaluate every model x training step
    # -----------------------------------------------------------------

    for model_id in model_ids:

        for step in steps:

            model_path = os.path.join(
                args.model_dir,
                f"{model_id}_ep{step}.ts",
            )

            if not os.path.exists(model_path):
                print(f"[WARNING] Checkpoint not found, skipping: {model_path}")
                continue

            print(ml_utils.format_header(70, f"{model_id} @ step {step}"))

            try:

                metrics = evaluate_model(
                    model_path=model_path,
                    builder=builder,
                    args=args,
                    latent_compactness=args.latent_compactness,
                    latent_mode=args.latent_mode,
                    max_iter=args.max_iter,
                )

            except Exception as e:

                print(
                    f"[ERROR] Failed to evaluate "
                    f"{model_path}: {e}"
                )

                continue

            results[model_id][step] = metrics

            rows.append(
                {
                    "model_id": model_id,
                    "step": step,
                    "balanced_accuracy": metrics[
                        "balanced_accuracy"
                    ],
                    "macro_f1": metrics[
                        "macro_f1"
                    ],
                }
            )

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------

    df = pd.DataFrame(rows)

    csv_path = os.path.join(
        args.output_dir,
        "latent_classification_over_training.csv",
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Display results
    # -----------------------------------------------------------------

    print(ml_utils.format_header(70, "Latent classification results"))
    print(df)

    print(f"\nSaved metrics table: {csv_path}")

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------

    metrics_to_plot = [
        "balanced_accuracy",
        "macro_f1",
    ]

    for metric in metrics_to_plot:

        output_path = os.path.join(args.output_dir, f"{metric}_over_training.png")

        plot_metric(
            results=results,
            model_ids=model_ids,
            steps=steps,
            metric=metric,
            output_path=output_path,
        )

        print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    _main()
