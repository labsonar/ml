import os
import argparse
import typing

import numpy as np
import pandas as pd
import torch
import torch.utils.data as torch_data

import lps_ml.utils.device as ml_device
import lps_ml.model as ml_model
import lps_ml.datasets as ml_db
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv
import lps_ml.utils.general as ml_utils

def latent_distance(x1, x2):
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=[1, 2]) + 1e-8)

def compute_scores(W_ct, W_gt, W_cg):
    eps = 1e-8

    # S1: proximidade ao target
    S1 = 1.0 - (W_gt / (W_ct + eps))

    # S2: consistência geométrica
    S2 = W_ct / (W_cg + W_gt + eps)

    return S1, S2


def evaluate_loader(model, loader, device):
    model.eval()

    S1_all = []
    S2_all = []

    with torch.no_grad():
        for batch in loader:
            x1, x2 = batch  # condicionante, target

            x1 = x1.to(device)
            x2 = x2.to(device)

            z_gen = model.sample(cond=x1)

            W_ct = latent_distance(x1, x2)
            W_gt = latent_distance(z_gen, x2)
            W_cg = latent_distance(x1, z_gen)

            S1, S2 = compute_scores(W_ct, W_gt, W_cg)

            S1_all.append(S1.cpu().numpy())
            S2_all.append(S2.cpu().numpy())

    S1_all = np.concatenate(S1_all)
    S2_all = np.concatenate(S2_all)

    return S1_all, S2_all


def summarize(values: np.ndarray):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--vae-model", type=str, default="/data/models/v0_6M.ts")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=str, default="./wasserstein_metrics.csv")

    args = parser.parse_args()

    ml_utils.set_seed()

    device = ml_device.get_available_device()

    vae_encoder = ml_procs.VAEEncoder(args.vae_model)

    n_samples = int(2**17)
    overlap = int(2**16)
    latent_compactness = int(2**10)

    dm = ml_db.IemanjaPaired(
        file_processor=ml_procs.SampleProcessor(
            n_samples=int(n_samples / latent_compactness),
            overlap=int(overlap / latent_compactness),
            pipelines=[
                ml_procs.ToFloatConverter(),
                vae_encoder
            ]
        ),
        cv=ml_cv.SimpleSplitCV(),
        dynamic_selection=ml_db.DynamicSelection.FIXED_ONLY,
        channel_selection=ml_db.ChannelSelection.REFERENCE_ONLY,
        batch_size=args.batch_size
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    x1, _ = next(iter(train_loader))

    latent_channels = x1.shape[1]

    model = ml_model.LatentDiffusionModel.load_from_checkpoint(
        args.model_checkpoint,
    )
    model.to(device)

    results = []

    splits = {
        "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
        "test": dm.test_dataloader(),
    }

    for split_name, loader in splits.items():
        print(f"Evaluating {split_name}...")

        S1, S2 = evaluate_loader(model, loader, device)

        stats_S1 = summarize(S1)
        stats_S2 = summarize(S2)

        for metric_name, stats in zip(["S1", "S2"], [stats_S1, stats_S2]):
            row = {
                "split": split_name,
                "metric": metric_name,
                **stats
            }
            results.append(row)

    df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)

    print("\nResultado final:")
    print(df)


if __name__ == "__main__":
    main()
