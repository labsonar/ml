"""
Evaluate a trained Latent Diffusion Model (LDM) over the validation set.
"""

import os
import shutil
import argparse
import collections
import numpy as np
import scipy.linalg as sci_alg
import scipy.spatial.distance as sci_dist
import sklearn.decomposition as skl_dec
import ot

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



def main():

    builder = ml_db.IemanjaBuilder(vae_exclusive=True)

    parser = argparse.ArgumentParser(
        description="Evaluate LDM latent reconstructions."
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

    psd_dir = os.path.join(args.output_dir, "psd")
    tsne_dir = os.path.join(args.output_dir, "tsne")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(psd_dir, exist_ok=True)
    os.makedirs(tsne_dir, exist_ok=True)

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

    metrics = collections.defaultdict(list)

    all_latent = []
    all_psd = []
    all_labels = []

    sample_metrics = []

    with torch.no_grad():

        global_sample_id = 0

        for _, batch in enumerate(val_loader):

            x_cond, x_target = batch
            x_cond = x_cond.to(device)
            x_target = x_target.to(device)

            x_generated = model.sample(cond=x_cond)

            t_cond = vae_encoder.decode(x_cond)
            t_target = vae_encoder.decode(x_target)
            t_generated = vae_encoder.decode(x_generated)

            for i in range(x_cond.shape[0]):

                cond_points = x_cond[i].detach().cpu().numpy().T
                target_points = x_target[i].detach().cpu().numpy().T
                generated_points = x_generated[i].detach().cpu().numpy().T

                t_cond_points = t_cond[i].detach().cpu().numpy()
                t_target_points = t_target[i].detach().cpu().numpy()
                t_generated_points = t_generated[i].detach().cpu().numpy()

                _, psd_cond = lps_bb.psd(t_cond_points, fs, window_size=4096, overlap=0.5)
                _, psd_target = lps_bb.psd(t_target_points, fs, window_size=4096, overlap=0.5)
                _, psd_generated = lps_bb.psd(t_generated_points, fs, window_size=4096, overlap=0.5)

                all_latent.append(cond_points.reshape(-1))
                all_psd.append(psd_cond)
                all_labels.append("Conditioning")

                all_latent.append(target_points.reshape(-1))
                all_psd.append(psd_target)
                all_labels.append("Target")

                all_latent.append(generated_points.reshape(-1))
                all_psd.append(psd_generated)
                all_labels.append("Generated")

                n_components=1
                pca_cond = skl_dec.PCA(n_components=n_components).fit(cond_points).components_
                pca_target = skl_dec.PCA(n_components=n_components).fit(target_points).components_
                pca_generated = skl_dec.PCA(n_components=n_components).fit(generated_points).components_

                ### ========= Plot PSDs ========= ###
                lps_bb.plot_psds(
                    filename=os.path.join(psd_dir, f"psd_{global_sample_id:06d}.png"),
                    fs=fs,
                    noises=[t_cond_points, t_target_points, t_generated_points],
                    labels=["Conditioning", "Target", "Generated"],
                    window_size=4096,
                    overlap=0.5
                )

                ### ========= MSE ========= ###
                latent_dist_cg = mse_similarity(cond_points, generated_points)
                latent_dist_tg = mse_similarity(target_points, generated_points)
                latent_dist_ct = mse_similarity(target_points, cond_points)

                latent_dist_align = 1 - latent_dist_tg/latent_dist_ct
                latent_dist_prox = latent_dist_cg/latent_dist_tg

                metrics["latent_dist_cg"].append(latent_dist_cg)
                metrics["latent_dist_tg"].append(latent_dist_tg)
                metrics["latent_dist_ct"].append(latent_dist_ct)
                metrics["latent_dist_align"].append(latent_dist_align)
                metrics["latent_dist_prox"].append(latent_dist_prox)

                ## ========= FID ========= ###
                fid_cg = latent_fid(cond_points, generated_points)
                fid_tg = latent_fid(target_points, generated_points)
                fid_ct = latent_fid(target_points, cond_points)

                fid_align = 1 - fid_tg/fid_ct
                fid_prox = fid_cg/fid_tg

                metrics["fid_cg"].append(fid_cg)
                metrics["fid_tg"].append(fid_tg)
                metrics["fid_ct"].append(fid_ct)
                metrics["fid_align"].append(fid_align)
                metrics["fid_prox"].append(fid_prox)

                sample_metrics.append({
                    "global_id": global_sample_id,
                    "ord": fid_prox,
                })

                ## ========= Wassertein ========= ###
                wass_cg = wasserstein(cond_points, generated_points)
                wass_tg = wasserstein(target_points, generated_points)
                wass_ct = wasserstein(target_points, cond_points)

                wass_align = 1 - wass_tg/wass_ct
                wass_prox = wass_cg/wass_tg

                metrics["wass_cg"].append(wass_cg)
                metrics["wass_tg"].append(wass_tg)
                metrics["wass_ct"].append(wass_ct)
                metrics["wass_align"].append(wass_align)
                metrics["wass_prox"].append(wass_prox)

                ## ========= PCA  ========= ###
                pca_cg = np.mean(np.cos(sci_alg.subspace_angles(pca_cond.T, pca_generated.T)))
                pca_tg = np.mean(np.cos(sci_alg.subspace_angles(pca_target.T, pca_generated.T)))
                pca_ct = np.mean(np.cos(sci_alg.subspace_angles(pca_cond.T, pca_target.T)))

                pca_align = 1 - pca_tg/pca_ct
                pca_prox = pca_cg/pca_tg

                metrics["pca_cg"].append(pca_cg)
                metrics["pca_tg"].append(pca_tg)
                metrics["pca_ct"].append(pca_ct)
                metrics["pca_align"].append(pca_align)
                metrics["pca_prox"].append(pca_prox)

                # ========= PSD MSE ========= ###
                psd_dist_cg = mse_similarity(psd_cond, psd_generated)
                psd_dist_tg = mse_similarity(psd_target, psd_generated)
                psd_dist_ct = mse_similarity(psd_target, psd_cond)

                psd_dist_align = 1 - psd_dist_tg/psd_dist_ct
                psd_dist_prox = psd_dist_cg/psd_dist_tg

                metrics["psd_dist_cg"].append(psd_dist_cg)
                metrics["psd_dist_tg"].append(psd_dist_tg)
                metrics["psd_dist_ct"].append(psd_dist_ct)
                metrics["psd_dist_align"].append(psd_dist_align)
                metrics["psd_dist_prox"].append(psd_dist_prox)


                ### ========= PSD Corr ========= ###
                psd_corr_cg = np.corrcoef(psd_cond, psd_generated)[0,1]
                psd_corr_tg = np.corrcoef(psd_target, psd_generated)[0,1]
                psd_corr_ct = np.corrcoef(psd_cond, psd_target)[0,1]

                psd_corr_align = 1 - psd_corr_tg/psd_corr_ct
                psd_corr_prox = psd_corr_cg/psd_corr_tg

                metrics["psd_corr_cg"].append(psd_corr_cg)
                metrics["psd_corr_tg"].append(psd_corr_tg)
                metrics["psd_corr_ct"].append(psd_corr_ct)
                metrics["psd_corr_align"].append(psd_corr_align)
                metrics["psd_corr_prox"].append(psd_corr_prox)

                ### ========= PSD cosine dist ========= ###
                psd_dcos_cg = sci_dist.cosine(psd_cond, psd_generated)
                psd_dcos_tg = sci_dist.cosine(psd_target, psd_generated)
                psd_dcos_ct = sci_dist.cosine(psd_cond, psd_target)

                psd_dcos_align = 1 - psd_dcos_tg/psd_dcos_ct
                psd_dcos_prox = psd_dcos_cg/psd_dcos_tg

                metrics["psd_dcos_cg"].append(psd_dcos_cg)
                metrics["psd_dcos_tg"].append(psd_dcos_tg)
                metrics["psd_dcos_ct"].append(psd_dcos_ct)
                metrics["psd_dcos_align"].append(psd_dcos_align)
                metrics["psd_dcos_prox"].append(psd_dcos_prox)


                # ### ========= Transform cosine similarity ========= ###
                delta_ct = target_points - cond_points
                delta_cg = generated_points - cond_points

                transform_cos = np.mean([
                    1 - sci_dist.cosine(a, b) for a, b in zip(delta_ct, delta_cg)
                ])
                metrics["transform_cos"].append(transform_cos)

                sample_metrics.append({
                    "global_id": global_sample_id,
                    "ord": transform_cos,
                })


                # ### ========= t-SNE latent ========= ###
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

                global_sample_id += 1

            #     break
            # break

    print("")

    for name, values in metrics.items():

        values = np.asarray(values)

        print(
            f"{name:10s}: "
            f"median={np.median(values): .6f} | "
            f"mean={np.mean(values): .6f} | "
            f"std={np.std(values): .6f} | "
            f"max={np.max(values): .6f} | "
            f"min={np.min(values): .6f}"
        )


    all_latent = np.asarray(all_latent)
    all_psd = np.asarray(all_psd)
    all_labels = np.asarray(all_labels)

    ml_vis.export_tsne(
        data=all_psd,
        labels=all_labels,
        filename=os.path.join(args.output_dir, "tsne_psd.png")
    )
    ml_vis.export_tsne(
        data=all_latent,
        labels=all_labels,
        filename=os.path.join(args.output_dir, "tsne_latent.png")
    )

    sample_metrics_sorted = sorted(
        sample_metrics,
        key=lambda x: x["ord"],
        reverse=True
    )

    ordered_dir = os.path.join(args.output_dir, "ordered")
    os.makedirs(ordered_dir, exist_ok=True)

    for i, item in enumerate(sample_metrics_sorted):

        global_id = item["global_id"]
        ord = item["ord"]

        # in_filename = os.path.join(tsne_dir, f"sample_{global_id:06d}_tsne.png")
        in_filename = os.path.join(psd_dir, f"psd_{global_id:06d}.png")
        out_filename = os.path.join(ordered_dir, f"{i}_{ord}_{global_id:06d}.png")

        shutil.copy2(in_filename, out_filename)

if __name__ == "__main__":
    main()
