"""
Evaluate a trained Latent Diffusion Model (LDM) over the validation set.
"""

import os
import shutil
import argparse
import collections
import numpy as np
import pandas as pd
import scipy.linalg as sci_alg
import scipy.spatial.distance as sci_dist
import sklearn.decomposition as skl_dec
import ot
import pickle
import matplotlib.pyplot as plt

import torch

import lps_ml.audio_processors as ml_procs
import lps_ml.datasets as ml_db
import lps_ml.core.cv as ml_cv
import lps_ml.utils.general as ml_utils
import lps_ml.utils.device as ml_device
import lps_ml.model as ml_model
import lps_ml.visualization.tsne as ml_vis
import lps_sp.acoustical.broadband as lps_bb
import lps_utils.quantities as lps_qty

def wasserstein(points_a: np.ndarray, points_b: np.ndarray) -> float:

    n_a = len(points_a)
    n_b = len(points_b)

    a = np.ones((n_a,)) / n_a
    b = np.ones((n_b,)) / n_b

    M = ot.dist(points_a, points_b, metric='sqeuclidean')

    wasserstein_sq = ot.emd2(a, b, M)

    return float(np.sqrt(wasserstein_sq))

def latent_fid(points_a: np.ndarray, points_b: np.ndarray) -> float:

    mu_a = np.mean(points_a, axis=0)
    mu_b = np.mean(points_b, axis=0)

    cov_a = np.cov(points_a, rowvar=False)
    cov_b = np.cov(points_b, rowvar=False)

    covmean = sci_alg.sqrtm(cov_a @ cov_b)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu_a - mu_b

    fid = diff @ diff + np.trace(cov_a + cov_b - 2 * covmean)

    return float(fid)

def mse_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes MSE between latent tensors.
    """
    return ((a - b) ** 2).mean()

def export_umap_projection(
    reducer,
    cond_points,
    target_points,
    generated_points,
    filename
):

    cond_umap = reducer.transform(cond_points)
    target_umap = reducer.transform(target_points)
    generated_umap = reducer.transform(generated_points)

    plt.figure(figsize=(8, 8))

    plt.scatter(
        cond_umap[:, 0],
        cond_umap[:, 1],
        s=10,
        alpha=0.7,
        label="Conditioning"
    )

    plt.scatter(
        target_umap[:, 0],
        target_umap[:, 1],
        s=10,
        alpha=0.7,
        label="Target"
    )

    plt.scatter(
        generated_umap[:, 0],
        generated_umap[:, 1],
        s=10,
        alpha=0.7,
        label="Generated"
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():

    builder = ml_db.IemanjaBuilder(ldm_exclusive=True)

    parser = argparse.ArgumentParser(
        description="Evaluate LDM latent reconstructions."
    )

    parser.add_argument(
        "--tsne-samples",
        type=int,
        default=20
    )

    parser.add_argument(
        "--umap-model",
        type=str,
        default=None
    )

    parser.add_argument(
        "--ldm-checkpoint",
        type=str,
        required=True,
        help="Path to LDM checkpoint (.ckpt)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./result/ldm_eval"
    )

    builder.add_argparse_args(parser=parser)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_float32_matmul_precision("medium")
    ml_utils.set_seed()

    device = ml_device.get_available_device()

    fs = lps_qty.Frequency.khz(16)


    dm = builder.paired_from_argparse_args(args)
    vae_encoder = dm.file_processor.pipelines[-1]
    dm.setup()

    val_loader = dm.val_dataloader()

    model = ml_model.LatentDiffusionModel.load_from_checkpoint(
        checkpoint_path=args.ldm_checkpoint
    )

    model.eval()
    model.to(device)

    if model.channel_mode == ml_model.ChannelMode.FIXED:
        channel_pairs = [
            (
                model.fixed_input_channel,
                model.fixed_output_channel
            )
        ]

    else:
        channel_pairs = [
            (i, j)
            for i in range(model.n_channels)
            for j in range(model.n_channels)
            if i != j
        ]

    metrics = collections.defaultdict(list)

    umap_model = None
    if args.umap_model is not None:
        with open(args.umap_model, "rb") as f:
            umap_model = pickle.load(f)

    with torch.no_grad():

        for in_ch, out_ch in channel_pairs:

            pair_name = f"{in_ch}->{out_ch}"

            all_cond_latent = []
            all_target_latent = []
            all_generated_latent = []
            sample_metrics = []

            if len(channel_pairs) == 1:
                psd_dir = os.path.join(args.output_dir, "psd")
                tsne_dir = os.path.join(args.output_dir, "tsne")
                umap_dir = os.path.join(args.output_dir, "umap")
                os.makedirs(psd_dir, exist_ok=True)
                os.makedirs(tsne_dir, exist_ok=True)
                os.makedirs(umap_dir, exist_ok=True)
            else:
                psd_dir = os.path.join(args.output_dir, "psd", pair_name)
                tsne_dir = os.path.join(args.output_dir, "tsne", pair_name)
                umap_dir = os.path.join(args.output_dir, "umap", pair_name)
                umaps_dir = os.path.join(args.output_dir, "umaps")
                os.makedirs(psd_dir, exist_ok=True)
                os.makedirs(tsne_dir, exist_ok=True)
                os.makedirs(umap_dir, exist_ok=True)
                os.makedirs(umaps_dir, exist_ok=True)

            global_sample_id = 0

            for _, (batch, target) in enumerate(val_loader):

                x_cond = batch[in_ch]
                x_target = batch[out_ch]

                x_cond = x_cond.to(device)
                x_target = x_target.to(device)
                target = target.to(device)

                x_generated = model.sample(cond=x_cond,
                                           distance=target,
                                           input_ch=in_ch,
                                           output_ch=out_ch)

                for i in range(x_cond.shape[0]):

                    cond_points = x_cond[i].detach().cpu().numpy().T
                    target_points = x_target[i].detach().cpu().numpy().T
                    generated_points = x_generated[i].detach().cpu().numpy().T

                    all_cond_latent.append(cond_points.T.reshape(-1))
                    all_target_latent.append(target_points.T.reshape(-1))
                    all_generated_latent.append(generated_points.T.reshape(-1))

                    ## ========= FID ========= ###
                    fid_cg = latent_fid(cond_points, generated_points)
                    fid_tg = latent_fid(target_points, generated_points)
                    fid_ct = latent_fid(target_points, cond_points)

                    fid_align = 1 - fid_tg/fid_ct
                    fid_prox = fid_cg/fid_tg

                    metrics[f"{pair_name}/fid_cg"].append(fid_cg)
                    metrics[f"{pair_name}/fid_tg"].append(fid_tg)
                    metrics[f"{pair_name}/fid_ct"].append(fid_ct)
                    metrics[f"{pair_name}/fid_align"].append(fid_align)
                    metrics[f"{pair_name}/fid_prox"].append(fid_prox)

                    sample_metrics.append({
                        "global_id": global_sample_id,
                        "ord": fid_prox,
                    })

                    # ### ========= Transform cosine similarity ========= ###
                    delta_ct = target_points - cond_points
                    delta_cg = generated_points - cond_points

                    transform_cos = np.mean([
                        1 - sci_dist.cosine(a, b) for a, b in zip(delta_ct, delta_cg)
                    ])
                    metrics[f"{pair_name}/transform_cos"].append(transform_cos)
                    metrics[f"{pair_name}/transform_angle"].append(np.degrees(np.arccos(transform_cos)))

                    sample_metrics.append({
                        "global_id": global_sample_id,
                        "ord": transform_cos,
                    })

                    # ### ========= t-SNE latent ========= ###
                    if global_sample_id < args.tsne_samples:
                        tsne_data = np.concatenate(
                            [
                                cond_points,
                                target_points,
                                generated_points
                            ],
                            axis=0
                        )

                        tsne_labels = np.concatenate(
                            [
                                np.full(cond_points.shape[0], "Conditioning"),
                                np.full(target_points.shape[0], "Target"),
                                np.full(generated_points.shape[0], "Generated"),
                            ]
                        )

                        tsne_filename = os.path.join(
                            tsne_dir,
                            f"sample_{global_sample_id:06d}_tsne.png"
                        )

                        ml_vis.export_tsne(
                            data=tsne_data,
                            labels=tsne_labels,
                            filename=tsne_filename
                        )

                        ### ========= UMAP latent ========= ###

                        if umap_model is not None:
                            umap_filename = os.path.join(
                                umap_dir,
                                f"sample_{global_sample_id:06d}_umap.png"
                            )

                            export_umap_projection(
                                reducer=umap_model,
                                cond_points=cond_points.T.reshape(1, -1),
                                target_points=target_points.T.reshape(1, -1),
                                generated_points=generated_points.T.reshape(1, -1),
                                filename=umap_filename
                            )

                        global_sample_id += 1


            if umap_model is not None:

                all_cond_latent = np.asarray(all_cond_latent)
                all_target_latent = np.asarray(all_target_latent)
                all_generated_latent = np.asarray(all_generated_latent)

                print("all_cond_latent: ", all_cond_latent.shape)

                umap_filename = os.path.join(umap_dir, "complete.png")

                export_umap_projection(
                    reducer=umap_model,
                    cond_points=all_cond_latent,
                    target_points=all_target_latent,
                    generated_points=all_generated_latent,
                    filename=umap_filename
                )

                if len(channel_pairs) > 1:
                    shutil.copy(umap_filename, os.path.join(umaps_dir, f"{pair_name}.png"))


    print("")

    rows = []

    for name, values in metrics.items():

        values = np.asarray(values)

        rows.append({
            "metric": name,
            "median": np.median(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "max": np.max(values),
            "min": np.min(values),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("metric")

    print(df)

    df.to_csv(os.path.join(args.output_dir, "metrics_summary.csv"), index=False)


if __name__ == "__main__":
    main()
