#!/usr/bin/env python3

"""
Compile and analyze predictions from multiple specialist classifiers.

The input directory is searched recursively for files matching:

    *_predictions.csv

Each CSV is expected to contain:

    ship_target,ship_pred,
    channel_target,channel_pred,
    shallow_target,shallow_pred

The script:
    1. Concatenates all prediction CSVs.
    2. Saves the consolidated predictions.
    3. Computes Balanced Accuracy and Macro F1.
    4. Saves one confusion matrix per task.
    5. Saves the performance metrics to CSV.

Example:

    python analyze_predictions.py \
        --input_dir ./runs \
        --output_dir ./results
"""

import argparse
import glob
import os
import typing

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics as sk_metrics

import lps_ml.utils.metrics as ml_metrics



def calculate_performance(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate classification performance for all tasks."""

    result = {}

    tasks = {
        "ship": list(range(2)),
        "channel": list(range(4)),
        "shallow": list(range(2)),
    }

    for task, labels in tasks.items():

        target_column = f"{task}_target"
        pred_column = f"{task}_pred"

        if target_column not in df.columns or pred_column not in df.columns:
            print(
                f"Warning: columns for '{task}' not found. "
                f"Skipping this task."
            )
            continue

        # Remove invalid rows.
        valid = df[[target_column, pred_column]].dropna()

        y_true = valid[target_column].astype(int)
        y_pred = valid[pred_column].astype(int)

        acc, f1 = ml_metrics.calculate_classification_metrics(y_true, y_pred)

        result[task] = {
            "acc": acc,
            "f1": f1,
        }

    return pd.DataFrame(result)

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Compile and analyze specialist classifier predictions."
    )
    parser.add_argument("--input_dir", required=True,
        help="Directory containing *_predictions.csv files.")
    parser.add_argument("--output_dir", default="./results",
        help="Directory where results will be saved.")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    pattern = os.path.join(args.input_dir, "*_predictions.csv")
    filenames = sorted(glob.glob(pattern, recursive=True))

    if not filenames:
        raise FileNotFoundError(f"No '*_predictions.csv' files found in: {args.input_dir}")

    print(f"Found {len(filenames)} prediction files.")

    dataframes = []

    for filename in filenames:

        print(f"Reading: {filename}")

        df = pd.read_csv(filename)

        expected_columns = [
            "CATALOG_ID",
            "fragment_id",
            "row_id",
            "ship_target",
            "ship_pred",
            "channel_target",
            "channel_pred",
            "shallow_target",
            "shallow_pred",
        ]

        missing = [
            column
            for column in expected_columns
            if column not in df.columns
        ]

        if missing:
            print(
                f"Warning: skipping {filename}. "
                f"Missing columns: {missing}"
            )
            continue

        # Keep track of the source specialist.
        df["source"] = os.path.basename(filename)

        dataframes.append(df)

    if not dataframes:
        raise RuntimeError("No valid prediction files were found.")

    combined_df = pd.concat(dataframes, ignore_index=True)

    predictions_filename = os.path.join(args.output_dir, "predictions.csv")
    combined_df.to_csv(predictions_filename, index=False)

    print(f"\nConsolidated predictions saved to:\n  {predictions_filename}")


    performance_df = calculate_performance(combined_df)

    performance_filename = os.path.join(args.output_dir, "performance.csv")
    performance_df.to_csv(performance_filename)

    print(f"Performance saved to:\n  {performance_filename}")

    print("\nPerformance:")
    print(performance_df)


    tasks = {
        "ship": list(range(2)),
        "channel": list(range(4)),
        "shallow": list(range(2)),
    }

    for task, labels in tasks.items():

        target_column = f"{task}_target"
        pred_column = f"{task}_pred"

        if (
            target_column not in combined_df.columns
            or pred_column not in combined_df.columns
        ):
            continue

        valid = combined_df[[target_column, pred_column]].dropna()

        y_true = valid[target_column].astype(int)
        y_pred = valid[pred_column].astype(int)

        filename = os.path.join(args.output_dir, f"{task}_confusion_matrix.png")

        ml_metrics.save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            filename=filename,
            labels=labels,
            title=f"{task.capitalize()} Class Confusion Matrix",
        )

        print(f"Confusion matrix saved to:\n  {filename}")


if __name__ == "__main__":
    main()
