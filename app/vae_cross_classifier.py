#!/usr/bin/env python3

"""
Cross-domain evaluation of two Iemanja classifiers.

The experiment compares:
    M1 = classifier trained on dataset 1
    M2 = classifier trained on dataset 2

Each classifier is evaluated on both datasets.

Datasets:
    dataset1 = original
    dataset2 = VAE reconstructed

For validation and test sets, the script computes:
    - Balanced Accuracy
    - Macro F1

Additionally, it computes classifier agreement:
    agreement = mean(M1_prediction == M2_prediction)

Agreement is computed independently for each dataset/split.

Outputs:

    output_dir/
    ├── metrics.csv
    ├── agreement.csv
    ├── cross_domain.csv
    ├── predictions.csv
    └── confusion/
        ├── M1_original_val.png
        ├── M1_VAE_val.png
        ├── M2_original_val.png
        ├── M2_VAE_val.png
        ├── ...
        └── ...
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.utils.data as torch_data

import sklearn.metrics as sk_metrics

import lps_ml.model as ml_model
import lps_ml.datasets as ml_db
import lps_ml.utils.device as ml_device
import lps_ml.utils.general as ml_utils

def load_model(path: str, model_type: str):
    """
    Load a CNN1D or CNN2D model from a checkpoint.
    """

    if os.path.isdir(path):
        path = os.path.join(path, "best.ckpt")

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    if model_type == "cnn1d":
        model_class = ml_model.CNN1D
    elif model_type == "cnn2d":
        model_class = ml_model.CNN2D
    else:
        raise ValueError(
            f"Unknown model type: {model_type}"
        )

    print(f"Loading model: {path}")
    print(f"Model type: {model_type}")

    model = model_class.load_from_checkpoint(path)
    model.eval()

    return model

def build_datamodule(builder, args, dataset_dir: str):
    """
    Build an Iemanja DataModule using the common processing arguments.
    """
    args.ie_dataset_dir = dataset_dir
    dm = builder.from_argparse_args(args)
    dm.setup()
    return dm

def get_split_ids(dm, split: str):
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

    return set(df["file_id"].dropna().astype(int).unique().tolist())

def verify_split_parity(
        dm1,
        dm2,
):
    """
    Verify that the two datasets have identical file IDs
    in train/validation/test.

    This is important because the VAE dataset is expected to be
    a transformed version of the original dataset.
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

            print(f"\nERROR: split '{split}' is not identical.")

            if only_1:
                print(f"  Only dataset1: {len(only_1)} files")

            if only_2:
                print(f"  Only dataset2: {len(only_2)} files")

            raise RuntimeError(
                "Dataset splits are not paired. "
                "Cross-domain comparison cannot be safely performed."
            )

    print("All splits have identical file IDs.")
    print("=" * 70)

def predict(
        model: torch.nn.Module,
        dataloader: torch_data.DataLoader,
        device,
):
    """
    Generate predictions and targets.

    Returns:
        y_true
        y_pred
    """

    model.eval()
    model.to(device)

    y_true = []
    y_pred = []

    with torch.inference_mode():

        for x, y in dataloader:

            # Paired datasets are not expected for this experiment.
            if isinstance(x, list):
                raise RuntimeError(
                    "Cross-domain classifier evaluation expects "
                    "a non-paired DataModule."
                )

            x = x.to(device)

            output = model(x)

            if output.ndim == 1:
                pred = (output >= 0.5).long()
            elif output.ndim == 2 and output.shape[1] == 1:
                pred = (output[:, 0] >= 0.5).long()
            else:
                pred = torch.argmax(output, dim=1)

            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(pred.detach().cpu().numpy())

    return np.asarray(y_true), np.asarray(y_pred)

def evaluate(
        model,
        dataloader,
        device,
):
    """
    Evaluate a classifier.
    """

    y_true, y_pred = predict(model=model, dataloader=dataloader, device=device)

    balanced_accuracy = sk_metrics.balanced_accuracy_score(y_true, y_pred)
    macro_f1 = sk_metrics.f1_score(y_true, y_pred, average="macro")

    return balanced_accuracy, macro_f1, y_true, y_pred

def save_confusion_matrix(
        y_true,
        y_pred,
        labels,
        filename,
        title,
):
    """
    Save a confusion matrix using seaborn.
    """

    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=labels,)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

def _main():

    parser = argparse.ArgumentParser(
        description=(
            "Cross-domain evaluation of two Iemanja classifiers."
        )
    )

    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help=(
            "Model 1 checkpoint or directory. "
            "If a directory is provided, best.ckpt is used."
        )
    )

    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help=(
            "Model 2 checkpoint or directory. "
            "If a directory is provided, best.ckpt is used."
        )
    )

    parser.add_argument(
        "--dataset1-dir",
        type=str,
        required=True,
        help="Dataset used to train model 1."
    )

    parser.add_argument(
        "--dataset2-dir",
        type=str,
        required=True,
        help="Dataset used to train model 2."
    )

    parser.add_argument(
        "--model1-type",
        choices=["cnn1d", "cnn2d"],
        default="cnn2d",
        help="Architecture used by model 1."
    )

    parser.add_argument(
        "--model2-type",
        choices=["cnn1d", "cnn2d"],
        default="cnn2d",
        help="Architecture used by model 2."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./result/classifier_comparison",
        help="Output directory."
    )

    builder = ml_db.IemanjaBuilder()
    builder.add_argparse_args(parser)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    ml_utils.set_seed()

    confusion_dir = os.path.join(args.output_dir, "confusion")
    os.makedirs(confusion_dir, exist_ok=True)

    device = ml_device.get_available_device()

    print()
    print("=" * 70)
    print("Iemanja classifier cross-domain evaluation")
    print("=" * 70)

    print(f"Model 1  : {args.model1}")
    print(f"Dataset 1: {args.dataset1_dir}")

    print()

    print(f"Model 2  : {args.model2}")
    print(f"Dataset 2: {args.dataset2_dir}")

    print()

    print(f"Device   : {device}")
    print(f"Output   : {args.output_dir}")

    print("=" * 70)

    model1 = load_model(args.model1, args.model1_type)
    model2 = load_model(args.model2, args.model2_type)


    print()
    print("Building dataset 1...")

    dm1 = build_datamodule(builder=builder, args=args, dataset_dir=args.dataset1_dir)

    print()
    print("Building dataset 2...")

    dm2 = build_datamodule(builder=builder, args=args, dataset_dir=args.dataset2_dir)

    verify_split_parity(dm1, dm2)

    dataset1_name = "original"
    dataset2_name = "reconstructed"

    model1_name = "M1"
    model2_name = "M2"

    metrics = []
    predictions = {}

    for split in ["val", "test"]:

        print()
        print("=" * 70)
        print(f"Evaluating {split}")
        print("=" * 70)

        if split == "val":
            loader1 = dm1.val_dataloader()
            loader2 = dm2.val_dataloader()

        else:
            loader1 = dm1.test_dataloader()
            loader2 = dm2.test_dataloader()

        evaluations = [
            (
                model1_name,
                dataset1_name,
                model1,
                loader1,
            ),
            (
                model1_name,
                dataset2_name,
                model1,
                loader2,
            ),
            (
                model2_name,
                dataset1_name,
                model2,
                loader1,
            ),
            (
                model2_name,
                dataset2_name,
                model2,
                loader2,
            ),
        ]

        for model_name, dataset_name, model, loader in evaluations:

            print(f"{model_name} -> {dataset_name} ({split})")

            balanced_accuracy, macro_f1, y_true, y_pred = evaluate(model=model,
                                                                   dataloader=loader,
                                                                   device=device)

            print(f"    BA      : {balanced_accuracy:.4f}")
            print(f"    Macro-F1: {macro_f1:.4f}")

            metrics.append(
                {
                    "split": split,
                    "model": model_name,
                    "dataset": dataset_name,
                    "balanced_accuracy": balanced_accuracy,
                    "macro_f1": macro_f1,
                }
            )

            predictions[(model_name, dataset_name, split)] = {
                    "y_true": y_true,
                    "y_pred": y_pred,
                }

            labels = sorted(np.unique(y_true))

            confusion_file = os.path.join(
                confusion_dir,
                f"{model_name}_{dataset_name}_{split}.png"
            )

            save_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                labels=labels,
                filename=confusion_file,
                title=(
                    f"{model_name} - "
                    f"{dataset_name} - "
                    f"{split}"
                )
            )

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)

    agreement_rows = []

    for split in ["val", "test"]:

        for dataset_name in [dataset1_name, dataset2_name]:

            p1 = predictions[
                (model1_name, dataset_name, split)
            ]

            p2 = predictions[
                (model2_name, dataset_name, split)
            ]

            if not np.array_equal(
                p1["y_true"],
                p2["y_true"]
            ):
                raise RuntimeError(
                    f"Ground truth mismatch for "
                    f"{dataset_name}/{split}."
                )

            if len(p1["y_pred"]) != len(p2["y_pred"]):
                raise RuntimeError(
                    f"Prediction size mismatch for "
                    f"{dataset_name}/{split}."
                )

            agreement = np.mean(p1["y_pred"] == p2["y_pred"])

            agreement_rows.append(
                {
                    "split": split,
                    "dataset": dataset_name,
                    "agreement": agreement,
                }
            )

    agreement_df = pd.DataFrame(agreement_rows)
    agreement_df.to_csv(os.path.join(args.output_dir, "agreement.csv"), index=False)

    cross_domain_rows = []

    for split in ["val", "test"]:

        m1_in = metrics_df[
            (metrics_df["model"] == "M1") &
            (metrics_df["dataset"] == dataset1_name) &
            (metrics_df["split"] == split)
        ].iloc[0]

        m1_cross = metrics_df[
            (metrics_df["model"] == "M1") &
            (metrics_df["dataset"] == dataset2_name) &
            (metrics_df["split"] == split)
        ].iloc[0]

        cross_domain_rows.append(
            {
                "split": split,
                "model": "M1",
                "in_domain": dataset1_name,
                "cross_domain": dataset2_name,

                "in_domain_balanced_accuracy":
                    m1_in["balanced_accuracy"],

                "cross_domain_balanced_accuracy":
                    m1_cross["balanced_accuracy"],

                "delta_balanced_accuracy":
                    m1_cross["balanced_accuracy"]
                    -
                    m1_in["balanced_accuracy"],

                "in_domain_macro_f1":
                    m1_in["macro_f1"],

                "cross_domain_macro_f1":
                    m1_cross["macro_f1"],

                "delta_macro_f1":
                    m1_cross["macro_f1"]
                    -
                    m1_in["macro_f1"],
            }
        )

        m2_in = metrics_df[
            (metrics_df["model"] == "M2") &
            (metrics_df["dataset"] == dataset2_name) &
            (metrics_df["split"] == split)
        ].iloc[0]

        m2_cross = metrics_df[
            (metrics_df["model"] == "M2") &
            (metrics_df["dataset"] == dataset1_name) &
            (metrics_df["split"] == split)
        ].iloc[0]

        cross_domain_rows.append(
            {
                "split": split,
                "model": "M2",
                "in_domain": dataset2_name,
                "cross_domain": dataset1_name,

                "in_domain_balanced_accuracy":
                    m2_in["balanced_accuracy"],

                "cross_domain_balanced_accuracy":
                    m2_cross["balanced_accuracy"],

                "delta_balanced_accuracy":
                    m2_cross["balanced_accuracy"]
                    -
                    m2_in["balanced_accuracy"],

                "in_domain_macro_f1":
                    m2_in["macro_f1"],

                "cross_domain_macro_f1":
                    m2_cross["macro_f1"],

                "delta_macro_f1":
                    m2_cross["macro_f1"]
                    -
                    m2_in["macro_f1"],
            }
        )

    cross_domain_df = pd.DataFrame(cross_domain_rows)
    cross_domain_df.to_csv(os.path.join(args.output_dir, "cross_domain.csv"), index=False)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("METRICS")
    print("=" * 70)
    print(metrics_df.to_string(index=False))

    print()
    print("=" * 70)
    print("AGREEMENT")
    print("=" * 70)
    print(agreement_df.to_string(index=False))

    print()
    print("=" * 70)
    print("CROSS-DOMAIN")
    print("=" * 70)
    print(cross_domain_df.to_string(index=False))

    print()
    print("=" * 70)
    print("Output")
    print("=" * 70)
    print(f"Metrics       : {args.output_dir}/metrics.csv")
    print(f"Agreement     : {args.output_dir}/agreement.csv")
    print(f"Cross-domain  : {args.output_dir}/cross_domain.csv")
    print(f"Confusion     : {confusion_dir}/")
    print("=" * 70)

if __name__ == "__main__":
    _main()
