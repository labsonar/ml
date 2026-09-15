#!/usr/bin/env python3
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


IDS = ["010M", "020M", "030M", "040M", "050M", "060M", "070M", "080M", "090M", "100M"]

def load_metrics(base_dir: str, model_names):
    records = []

    for model_name in model_names:
        for model_id in IDS:
            metrics_path = os.path.join(
                base_dir,
                f"{model_name}{model_id}",
                "spectral",
                "channel",
                "metrics.csv",
            )

            if not os.path.exists(metrics_path):
                print(f"[WARNING] File not found: {metrics_path}")
                continue

            print(f"[INFO] Reading: {metrics_path}")

            df = pd.read_csv(metrics_path)

            df = df[df["split"] == "val"].copy()

            if df.empty:
                print(f"[WARNING] No validation data: {metrics_path}")
                continue

            # Add experiment information
            df.insert(0, "model_name", model_name)
            df.insert(1, "ID", model_id)

            records.append(df)

    if not records:
        raise RuntimeError("No metrics.csv files were found.")

    return pd.concat(records, ignore_index=True)

def plot_metric(
    df,
    model_column,
    dataset,
    metric,
    ylabel,
    title,
    output_path,
):
    data = df[
        (df["model"] == model_column)
        & (df["dataset"] == dataset)
    ].copy()

    if data.empty:
        print(
            f"[WARNING] No data for "
            f"model={model_column}, dataset={dataset}, metric={metric}"
        )
        return

    data["epoch"] = data["ID"].str.replace("M", "", regex=False).astype(int)
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, group in data.groupby("model_name"):
        group = group.sort_values("epoch")

        ax.plot(
            group["epoch"],
            group[metric],
            marker="o",
            label=model_name,
        )

    ax.set_xlabel("Training epochs (×1000)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.set_xticks(range(10, 101, 10))
    ax.set_xticklabels([f"{x:03d}M" for x in range(10, 101, 10)])

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"[INFO] Saved: {output_path}")

def _main():
    parser = argparse.ArgumentParser(
        description=(
            "Compile channel classification metrics from multiple "
            "models and training epochs."
        )
    )

    parser.add_argument(
        "base_dir",
        type=str,
        help="Base directory containing the experiment directories.",
    )

    parser.add_argument(
        "models",
        nargs="+",
        help="Model names, e.g. modelA modelB modelC.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to base_dir/compiled_metrics.",
    )

    args = parser.parse_args()

    base_dir = args.base_dir

    if args.output_dir is None:
        output_dir = os.path.join(base_dir, "train_stop")
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    df = load_metrics(base_dir, args.models)

    df = df[
        [
            "model_name",
            "ID",
            "model",
            "dataset",
            "acc",
            "f1",
        ]
    ]

    df["ID_numeric"] = df["ID"].str.replace("M", "", regex=False).astype(int)

    df = df.sort_values(["model_name", "ID_numeric", "model", "dataset"]).drop(columns="ID_numeric")

    # Save compiled dataframe
    csv_path = os.path.join(output_dir, "metrics_compiled.csv")
    df.to_csv(csv_path, index=False)

    print(f"[INFO] Saved: {csv_path}")

    plot_metric(
        df=df,
        model_column="M1",
        dataset="reconstructed",
        metric="f1",
        ylabel="F1",
        title="Fidelity - F1",
        output_path=os.path.join(output_dir, "fidelity_f1.png"),
    )

    plot_metric(
        df=df,
        model_column="M1",
        dataset="reconstructed",
        metric="acc",
        ylabel="Accuracy",
        title="Fidelity - Accuracy",
        output_path=os.path.join(output_dir, "fidelity_acc.png"),
    )

    plot_metric(
        df=df,
        model_column="M2",
        dataset="original",
        metric="f1",
        ylabel="F1",
        title="Diversity - F1",
        output_path=os.path.join(output_dir, "diversity_f1.png"),
    )

    plot_metric(
        df=df,
        model_column="M2",
        dataset="original",
        metric="acc",
        ylabel="Accuracy",
        title="Diversity - Accuracy",
        output_path=os.path.join(output_dir, "diversity_acc.png"),
    )

    plot_metric(
        df=df,
        model_column="M2",
        dataset="reconstructed",
        metric="f1",
        ylabel="F1",
        title="Separability - F1",
        output_path=os.path.join(output_dir, "separability_f1.png"),
    )

    plot_metric(
        df=df,
        model_column="M2",
        dataset="reconstructed",
        metric="acc",
        ylabel="Accuracy",
        title="Separability - Accuracy",
        output_path=os.path.join(output_dir, "separability_acc.png"),
    )

if __name__ == "__main__":
    _main()
