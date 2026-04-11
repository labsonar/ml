import argparse
import os
import typing

import torch
import torch.utils.data as torch_data

import lps_utils.quantities as lps_qty
import lps_ml.datasets as ml_db
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv
import lps_ml.utils.separability as ml_sep


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data/models/v0_4M6.ts")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["SILHOUETTE", "DAVIES_BOULDIN"],
        choices=[m.name for m in ml_sep.Separability]
    )
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    metrics = [ml_sep.Separability[m].get() for m in args.metrics]

    fs = lps_qty.Frequency.khz(16)
    n_samples = int(2**17)
    overlap = int(2**16)
    latent_compactness = int(2**10)

    time_dm = ml_db.Iemanja(
            file_processor=ml_procs.SampleProcessor(
                    n_samples=n_samples,
                    overlap=overlap,
                    pipelines=[
                        ml_procs.ToFloatConverter(),
                    ]
                ),
            cv = ml_cv.SimpleSplitCV(),
            simple_version=True,
            batch_size=16
            )
    time_dm.setup()
    time_dict_loader = time_dm.all_dataloader_dict()

    time_res = ml_sep.SeparabilityMetric.compare_dataloaders(
        time_dict_loader[0],
        time_dict_loader[1],
        metrics
    )

    latent_dm = ml_db.Iemanja(
            file_processor=ml_procs.SampleProcessor(
                    n_samples=int(n_samples/latent_compactness),
                    overlap=int(overlap/latent_compactness),
                    pipelines=[
                        ml_procs.ToFloatConverter(),
                        ml_procs.VAEEncoder(args.model)
                    ]
                ),
            cv = ml_cv.FiveByTwo(),
            simple_version=True,
            batch_size=16
            )
    latent_dm.setup()
    latent_dict_loader = latent_dm.all_dataloader_dict()

    latent_res = ml_sep.SeparabilityMetric.compare_dataloaders(
        latent_dict_loader[0],
        latent_dict_loader[1],
        metrics
    )

    print("\n Separability Results:")
    print("time: ", time_res)
    print("latent: ", latent_res)



if __name__ == "__main__":
    main()
