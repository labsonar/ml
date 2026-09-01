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
import lps_ml.utils.default as ml_default
import lps_ml.visualization.umap as ml_umap

def _main():

    parser = argparse.ArgumentParser(
        description="Evaluate LDM latent reconstructions."
    )

    parser.add_argument("--test-samples", type=int, default=20)
    parser.add_argument("--umap-model", type=str, required=True)
    parser.add_argument("--ldm-checkpoint", type=str, required=True,
        help="Path to LDM checkpoint (.ckpt)")
    parser.add_argument("--output-dir", type=str, default="./result/ldm_eval")

    builder = ml_db.IemanjaBuilder(ldm_exclusive=True)
    builder.add_argparse_args(parser=parser)

    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.set_float32_matmul_precision("medium")
    ml_utils.set_seed()

    device = ml_device.get_available_device()

    dm = builder.paired_from_argparse_args(args)
    dm.setup()
    val_loader = dm.val_dataloader()

    ckpt = args.ldm_checkpoint
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, "best.ckpt")
    model = ml_model.LDM.load_from_checkpoint(checkpoint_path=ckpt)
    model.to(device)
    model.eval()

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

    umap_model = ml_umap.load_umap(args.umap_model)

    with torch.no_grad():

        for in_ch, out_ch in channel_pairs:

            pair_name = f"{in_ch}->{out_ch}"

            all_cond_latent = []
            all_target_latent = []
            all_generated_latent = []

            tsne_dir = os.path.join(args.output_dir, "tsne")
            umap_dir = os.path.join(args.output_dir, "umap")
            os.makedirs(tsne_dir, exist_ok=True)
            os.makedirs(umap_dir, exist_ok=True)

            if len(channel_pairs) > 1:
                umaps_dir = os.path.join(args.output_dir, "umaps")
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

                    # ### ========= t-SNE latent ========= ###
                    if global_sample_id < args.test_samples:
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

                            groups = {
                                "Conditioning": umap_model.transform(cond_points.T.reshape(1,-1)),
                                "Target": umap_model.transform(target_points.T.reshape(1,-1)),
                                "Generated": umap_model.transform(generated_points.T.reshape(1,-1))
                            }

                            umap_filename = os.path.join(
                                umap_dir,
                                f"sample_{global_sample_id:06d}_umap.png"
                            )

                            ml_umap.plot_2d_embedding(groups = groups, filename = umap_filename)

                        global_sample_id += 1


            if umap_model is not None:

                all_cond_latent = np.asarray(all_cond_latent)
                all_target_latent = np.asarray(all_target_latent)
                all_generated_latent = np.asarray(all_generated_latent)

                if len(channel_pairs) > 1:
                    umap_filename = os.path.join(umap_dir,  f"{pair_name}.png")
                else:
                    umap_filename = os.path.join(umap_dir, "complete.png")

                groups = {
                    "Conditioning": umap_model.transform(all_cond_latent),
                    "Target": umap_model.transform(all_target_latent),
                    "Generated": umap_model.transform(all_generated_latent)
                }

                ml_umap.plot_2d_embedding(groups = groups, filename = umap_filename)


if __name__ == "__main__":
    _main()
