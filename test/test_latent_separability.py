import argparse
import os
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import umap
import random

import torch
import torch.utils.data as torch_data

import lps_utils.quantities as lps_qty
import lps_ml.datasets as ml_db
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv
import lps_ml.visualization.separability as ml_sep
import lps_ml.utils.general as ml_utils
import lps_ml.visualization.tsne as ml_vis

def _extract_latent_and_labels(loader, latent_mode: str = "flatten"):

    all_data = []
    all_labels = []

    for x, y in loader:

        if not all_data:
            print("\tshape: ", x.shape)

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

    data = np.vstack(all_data)
    labels = np.concatenate(all_labels)

    return data, labels

def export_umap(
    data: np.ndarray,
    labels: np.ndarray,
    filename: str,
    model_filename: str,
    embedding_filename: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean"
):

    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=0
    )

    embedding = reducer.fit_transform(data)

    with open(model_filename, "wb") as f:
        pickle.dump(reducer, f)

    np.savez(
        embedding_filename,
        embedding=embedding,
        labels=labels
    )

    plt.figure(figsize=(8, 8))

    unique_labels = np.unique(labels)

    for label in unique_labels:

        mask = labels == label

        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=8,
            alpha=0.5,
            label=str(label)
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return reducer

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

def _parse_model_specs(model_specs: list[str], default_compactness: int) -> typing.Dict[str, int]:
    model_specs = [os.path.abspath(p) for p in model_specs]

    parsed = {}

    for spec in model_specs:
        if ":" in spec:
            try:
                path, comp = spec.rsplit(":", 1)
                compactness = int(comp)

            except ValueError:
                raise ValueError(
                    f"Invalid compactness value in '{spec}'. "
                    f"Use format model_path:latent_compactness"
                )

        else:
            path = spec
            compactness = default_compactness

        parsed[os.path.abspath(path)] = compactness

    return parsed

def _get_loader_dict(dm, fold_role: str):

    if fold_role is None:
        return dm.all_dataloader_dict()

    role = ml_cv.FoldRole[fold_role]

    if role == ml_cv.FoldRole.TRAIN:
        return dm.train_dataloader_dict()

    if role == ml_cv.FoldRole.VALIDATION:
        return dm.val_dataloader_dict()

    if role == ml_cv.FoldRole.TEST:
        return dm.test_dataloader_dict()

    raise ValueError(f"Unsupported fold role: {fold_role}")

def _get_loader(dm, fold_role: str):

    if fold_role is None:
        return dm.all_dataloader()

    role = ml_cv.FoldRole[fold_role]

    if role == ml_cv.FoldRole.TRAIN:
        return dm.train_dataloader()

    if role == ml_cv.FoldRole.VALIDATION:
        return dm.val_dataloader()

    if role == ml_cv.FoldRole.TEST:
        return dm.test_dataloader()

    raise ValueError(f"Unsupported fold role: {fold_role}")

def main():

    builder = ml_db.IemanjaBuilder(vae_exclusive=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model_path:latent_compactness"
    )
    parser.add_argument("--export_umap", action="store_true", help="Export umap")
    parser.add_argument("--default_compactness", type=int, default=1024)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["KNN"],
        choices=[m.name for m in ml_sep.Separability]
    )
    parser.add_argument("--fold_role", type=str, default=None,
        choices=[f.name for f in ml_cv.FoldRole]
    )
    parser.add_argument("--latent_mode", type=str, default="flatten",
        choices=["flatten", "samples"]
    )
    parser.add_argument("--output_dir", type=str, default="./result/latent_separability")
    builder.add_argparse_args(parser=parser)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    metrics = [ml_sep.Separability[m].get() for m in args.metrics]

    latent_results = {}

    compactness_dict = _parse_model_specs(args.models, args.default_compactness)
    models = compactness_dict.keys()

    for name, model_path in ml_utils.shortest_relative_path(models):

        try:
            latent_compactness = compactness_dict[model_path]

            latent_dm = builder.from_argparse_args(args, model_path, latent_compactness)
            latent_dm.batch_size = 1
            latent_dm.num_workers = 0
            latent_dm.setup()
            latent_dict_loader = _get_loader_dict(latent_dm, args.fold_role)

            latent_results[name] = ml_sep.SeparabilityMetric.compare_dataloaders(
                latent_dict_loader[0],
                latent_dict_loader[1],
                metrics
            )

            loader = _get_loader(latent_dm, args.fold_role)

            print("model: ", name)
            data, labels = _extract_latent_and_labels(loader, latent_mode=args.latent_mode)

            aux_name = name.replace("/", "_")
            filename = os.path.join(output_dir, f"tsne_{aux_name}.png")

            ml_vis.export_tsne(
                data=data,
                labels=labels,
                filename=filename
            )

            print(f"Saved t-SNE: {filename}")

            if args.export_umap:

                umap_plot = os.path.join(output_dir, f"umap_{aux_name}.png")
                umap_model = os.path.join(output_dir, f"umap_{aux_name}.pkl")
                umap_embedding = os.path.join(output_dir, f"umap_{aux_name}.npz")

                export_umap(
                    data=data,
                    labels=labels,
                    filename=umap_plot,
                    model_filename=umap_model,
                    embedding_filename=umap_embedding,
                    metric="cosine"
                )

                print(f"Saved UMAP: {umap_plot}")

        except Exception as e:
            print(f"Error processing {model_path}: {e}")

    df = pd.DataFrame.from_dict(latent_results, orient="index")
    df.to_csv(os.path.join(output_dir, "latent_separability.csv"), index=True)
    _save_metric_heatmaps(df, output_dir)

    print("\nLatent separability (DataFrame):")
    print(df)



if __name__ == "__main__":
    main()
