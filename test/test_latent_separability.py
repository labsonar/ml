import argparse
import os
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.utils.data as torch_data

import lps_utils.quantities as lps_qty
import lps_ml.datasets as ml_db
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv
import lps_ml.utils.separability as ml_sep
import lps_ml.utils.general as ml_utils
import lps_ml.visualization.tsne as ml_vis

def _extract_latent_and_labels(loader):

    all_data = []
    all_labels = []

    for x, y in loader:

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        all_data.append(x)
        all_labels.append(y)

    data = np.vstack(all_data)
    labels = np.concatenate(all_labels)

    return data, labels

def _save_metric_heatmaps(df, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    for metric in df.columns:

        metric_df = df[[metric]]

        plt.figure(figsize=(6, max(4, len(metric_df) * 0.5)))

        sns.heatmap(
            metric_df,
            annot=True,
            fmt=".3f",
            linewidths=0.5,
            cmap="viridis",
            cbar=True
        )

        plt.title(metric)
        plt.ylabel("Model")
        plt.xlabel("")

        plt.tight_layout()

        filename = os.path.join(output_dir, f"heatmap_{metric}.png")
        plt.savefig(filename, dpi=300)
        plt.close()

def _parse_model_specs(model_specs: list[str]) -> typing.Dict[str, int]:
    model_specs = [os.path.abspath(p) for p in model_specs]

    parsed = {}

    for spec in model_specs:
        try:
            path, comp = spec.split(":")
            parsed[path] = int(comp)

        except ValueError:
            raise ValueError(
                f"Invalid model specification '{spec}'. Use format model_path:latent_compactness"
            )

    return parsed

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model_path:latent_compactness"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["SILHOUETTE", "DAVIES_BOULDIN"],
        choices=[m.name for m in ml_sep.Separability]
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./result/latent_separability")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    metrics = [ml_sep.Separability[m].get() for m in args.metrics]

    n_samples = int(2**17)
    overlap = int(2**16)

    latent_results = {}

    compactness_dict = _parse_model_specs(args.models)
    models = compactness_dict.keys()

    for name, model_path in ml_utils.shortest_relative_path(models):
        print(f"Processing model: {model_path}")
        print("compactness_dict: ", compactness_dict)

        try:
            latent_compactness = compactness_dict[model_path]

            print("latent_compactness: ", latent_compactness)
            print("n_samples: ", n_samples)
            print("overlap: ", overlap)

            latent_dm = ml_db.Iemanja(
                    file_processor=ml_procs.SampleProcessor(
                            n_samples=int(n_samples/latent_compactness),
                            overlap=int(overlap/latent_compactness),
                            pipelines=[
                                ml_procs.ToFloatConverter(),
                                ml_procs.VAEEncoder(model_path)
                            ]
                        ),
                    cv = ml_cv.FiveByTwo(),
                    simple_version=True,
                    batch_size=args.batch_size,
                    )
            latent_dm.setup()
            latent_dict_loader = latent_dm.val_dataloader_dict()

            latent_results[name] = ml_sep.SeparabilityMetric.compare_dataloaders(
                latent_dict_loader[0],
                latent_dict_loader[1],
                metrics
            )

            loader = latent_dm.val_dataloader()

            data, labels = _extract_latent_and_labels(loader)

            aux_name = name.replace("/", "_")
            filename = os.path.join(output_dir, f"tsne_{aux_name}.png")

            ml_vis.export_tsne(
                data=data,
                labels=labels,
                filename=filename
            )

            print(f"Saved t-SNE: {filename}")

        except Exception as e:
            print(f"Error processing {model_path}: {e}")

    df = pd.DataFrame.from_dict(latent_results, orient="index")
    df.to_csv(os.path.join(output_dir, "latent_separability.csv"), index=True)
    _save_metric_heatmaps(df, output_dir)

    print("\nLatent separability (DataFrame):")
    print(df)



if __name__ == "__main__":
    main()
