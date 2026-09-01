#!/usr/bin/env python3
"""
Project a dataset and optionally a WAV file into a previously trained UMAP space.
"""

import argparse
import os
import pickle
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io.wavfile as scipy_wav
import torch

import lps_ml.utils.general as ml_utils
import lps_ml.utils.default as ml_default
import lps_ml.datasets as ml_db
import lps_ml.core.cv as ml_cv
import lps_ml.visualization.separability as ml_sep
import lps_ml.visualization.umap as ml_umap


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
        help="List of model_path:latent_compactness"
    )
    parser.add_argument("--default-compactness", type=int, default=1024)
    parser.add_argument("--export-umap", action="store_true", help="Export umap")
    parser.add_argument("--fold-role", type=str, default=None,
        choices=[f.name for f in ml_cv.FoldRole]
    )
    parser.add_argument("--latent-mode", type=str, default="flatten",
        choices=["flatten", "samples"]
    )
    parser.add_argument("--output-dir", type=str, default="./result/latent_separability")
    parser.add_argument("--umap-dir", required=True)
    parser.add_argument("--wav", default=None)

    builder = ml_db.IemanjaBuilder(vae_exclusive=True)
    builder.add_argparse_args(parser=parser)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results = []


    compactness_dict = ml_default.parse_model_specs(args.models, args.default_compactness)
    models = compactness_dict.keys()

    for name, model_path in ml_utils.shortest_relative_path(models):

        umap_path = ml_umap.umap_string(name, args.umap_dir)

        reducer = ml_umap.load_umap(umap_path)

        latent_compactness = compactness_dict[model_path]
        dm = builder.from_argparse_args(args, model_path, latent_compactness)
        dm.setup()

        loader = dm.get_dataloader_by_role(args.fold_role)

        print("Extracting dataset latents...")
        data, labels = ml_sep.extract_latent_and_labels(loader, latent_mode=args.latent_mode)

        if args.wav is not None:
            wav_data = ml_default.process_wav_to_latent(dm, args.wav, latent_mode=args.latent_mode)

            data = np.vstack([data, wav_data])
            wav_labels = np.full(len(wav_data), "wav_file")
            labels = np.concatenate([labels, wav_labels])

        print("Projecting dataset...")
        embedding_dataset = reducer.transform(data)

        groups = {
            str(label): embedding_dataset[labels == label]
            for label in np.unique(labels)
        }

        plot_path = ml_umap.umap_string(name, output_dir).replace(".pkl", ".png")
        ml_umap.plot_2d_embedding(
            groups=groups,
            filename=plot_path,
            highlight="wav_file"
        )

        knn = ml_sep.KNNConsistency(k=20)
        knn_umap = knn.compute(embedding_dataset, labels)

        results.append({
            "Model": name,
            "KNN_UMAP": knn_umap,
        })

    df = pd.DataFrame(results)
    df.set_index("Model", inplace=True)
    df.to_csv(os.path.join(output_dir, "knn_results.csv"))
    print(df)

if __name__ == "__main__":
    main()
