"""Vae Latent Analysis
"""
import os
import argparse
import pandas as pd

import lps_ml.datasets as ml_db
import lps_ml.core.cv as ml_cv
import lps_ml.visualization.tsne as ml_vis
import lps_ml.visualization.separability as ml_sep
import lps_ml.visualization.umap as ml_umap
import lps_ml.utils.general as ml_utils
import lps_ml.utils.default as ml_default


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

    builder = ml_db.IemanjaBuilder(vae_exclusive=True)
    builder.add_argparse_args(parser=parser)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    latent_results = {}

    compactness_dict = ml_default.parse_model_specs(args.models, args.default_compactness)
    models = compactness_dict.keys()

    for name, model_path in ml_utils.shortest_relative_path(models):

        try:
            latent_compactness = compactness_dict[model_path]

            latent_dm = builder.from_argparse_args(args, model_path, latent_compactness)
            latent_dm.setup()
            latent_dict_loader = latent_dm.get_dataloader_dict_by_role(args.fold_role)

            latent_results[name] = ml_sep.SeparabilityMetric.compare_dataloaders(
                latent_dict_loader[0],
                latent_dict_loader[1],
                [ml_sep.Separability.KNN]
            )

            loader = latent_dm.get_dataloader_by_role(args.fold_role)

            data, labels = ml_sep.extract_latent_and_labels(loader, latent_mode=args.latent_mode)

            aux_name = name.replace("/", "_")
            filename = os.path.join(output_dir, f"tsne_{aux_name}.png")

            ml_vis.export_tsne(data=data, labels=labels, filename=filename)

            print(f"Saved t-SNE: {filename}")

            if args.export_umap:

                umap_model = ml_umap.umap_string(model_name=name, output_dir=output_dir)
                umap_plot = umap_model.replace(".pkl", ".png")

                _, embedding, trust = ml_umap.export_umap(
                    data=data,
                    labels=labels,
                    filename=umap_plot,
                    model_filename=umap_model,
                    metric="cosine"
                )

                knn_metric = ml_sep.KNNConsistency(k=20)
                knn_umap = knn_metric.compute(embedding, labels)
                latent_results[name]["KNN_UMAP"] = knn_umap

                latent_results[name]["Trustworthiness"] = trust

                print(f"Saved UMAP: {umap_plot}")

        except Exception as e:
            print(f"Error processing {model_path}: {e}")

    df = pd.DataFrame.from_dict(latent_results, orient="index")
    df.to_csv(os.path.join(output_dir, "latent_separability.csv"), index=True)

    print("\nLatent separability (DataFrame):")
    print(df)



if __name__ == "__main__":
    main()
