import argparse
import os
import itertools
import typing

import numpy as np
import torch
import torch.utils.data as torch_data

import lps_utils.quantities as lps_qty

import lps_ml.datasets as ml_db
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv
import lps_ml.visualization.separability as ml_sep



def build_dataloader(
    input_dir: str,
    batch_size: int,
    num_workers: int
) -> torch_data.DataLoader:

    fs = lps_qty.Frequency.khz(16)
    n_samples = int(2**17)
    overlap = int(2**16)

    dm = ml_db.AudioFolder(
        input_dir=input_dir,
        file_processor=ml_procs.SampleProcessor(
            fs_out=fs,
            n_samples=n_samples,
            overlap=overlap,
            pipelines=[ml_procs.ToFloatConverter()]
        ),
        cv=ml_cv.SimpleSplitCV(),
        batch_size=batch_size,
        num_workers=num_workers
    )

    dm.prepare_data()
    dm.setup()

    return dm.all_dataloader(shuffle=False)


def load_all_datasets(
    input_dirs: typing.List[str],
    batch_size: int,
    num_workers: int
) -> dict:

    datasets = {}

    for path in input_dirs:
        name = os.path.basename(os.path.normpath(path))
        print(f"\n📂 Loading: {name}")

        loader = build_dataloader(path, batch_size, num_workers)

        data = ml_sep.dataloader_to_numpy(loader)
        datasets[name] = data

        print(f"   Shape: {data.shape}")

    return datasets


def pairwise_comparison(
    datasets: dict,
    metrics: typing.List[ml_sep.SeparabilityMetric]
) -> dict:

    results = {}

    pairs = list(itertools.combinations(datasets.keys(), 2))

    for a, b in pairs:
        print(f"\n🔍 Comparing: {a} vs {b}")

        res = ml_sep.SeparabilityMetric.compare_dataloaders(
            datasets[a],
            datasets[b],
            metrics
        )

        results[(a, b)] = res

    return results


def print_table(results: dict):
    print("\n📊 Pairwise Results:\n")

    for (a, b), metrics in results.items():
        print(f"{a} vs {b}")
        for m, v in metrics.items():
            print(f"  {m}: {v:.4f}")
        print()


def main():

    parser = argparse.ArgumentParser(
        description="Separability metrics for audio datasets"
    )

    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="List of dataset directories"
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[ml_sep.Separability.SILHOUETTE.name],
        choices=[m.name for m in ml_sep.Separability],
        help="Separability metrics"
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()

    metrics = [ml_sep.Separability[m].get() for m in args.metrics]

    datasets = load_all_datasets(
        args.input_dirs,
        args.batch_size,
        args.num_workers
    )

    if len(datasets) < 2:
        raise ValueError("Need at least 2 datasets")

    results = pairwise_comparison(datasets, metrics)

    print_table(results)


if __name__ == "__main__":
    main()