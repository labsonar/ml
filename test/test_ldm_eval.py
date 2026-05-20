"""
Evaluate a trained Latent Diffusion Model (LDM) over the validation set.
"""

import os
import argparse
import numpy as np
import scipy.linalg as sci_alg

import torch

import lps_ml.audio_processors as ml_procs
import lps_ml.datasets as ml_db
import lps_ml.core.cv as ml_cv
import lps_ml.utils.general as ml_utils
import lps_ml.utils.device as ml_device
import lps_ml.model as ml_model
import lps_ml.visualization.tsne as ml_vis

# def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     """
#     Computes cosine similarity between latent tensors.
#     """
#     a = a.flatten(start_dim=1)
#     b = b.flatten(start_dim=1)

#     return F.cosine_similarity(a, b, dim=1)


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

    parser = argparse.ArgumentParser(
        description="Evaluate LDM latent reconstructions."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to LDM checkpoint (.ckpt)"
    )

    parser.add_argument(
        "--vae-model",
        type=str,
        required=True,
        help="Path to VAE encoder model"
    )

    parser.add_argument(
        "--latent_compactness",
        type=int,
        default=1024
    )

    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training.")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./result/ldm_eval"
    )

    parser.add_argument(
        "--dynamic_selection",
        type=str,
        default=ml_db.DynamicSelection.FIXED_ONLY.name,
        choices=[e.name for e in ml_db.DynamicSelection],
    )

    parser.add_argument(
        "--channel_selection",
        type=str,
        default=ml_db.ChannelSelection.REFERENCE_ONLY.name,
        choices=[e.name for e in ml_db.ChannelSelection],
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_float32_matmul_precision("medium")
    ml_utils.set_seed()

    device = ml_device.get_available_device()

    dynamic_selection = ml_db.DynamicSelection[args.dynamic_selection]
    channel_selection = ml_db.ChannelSelection[args.channel_selection]

    n_samples = int(2**17)
    overlap = int(2**16)

    vae_encoder = ml_procs.VAEEncoder(args.vae_model)

    dm = ml_db.IemanjaPaired(
        file_processor=ml_procs.SampleProcessor(
            n_samples=int(n_samples / args.latent_compactness),
            overlap=int(overlap / args.latent_compactness),
            pipelines=[
                ml_procs.ToFloatConverter(),
                vae_encoder
            ]
        ),
        cv=ml_cv.SimpleSplitCV(),
        dynamic_selection=dynamic_selection,
        channel_selection=channel_selection,
        batch_size=args.batch_size
    )

    dm.setup()

    val_loader = dm.val_dataloader()

    model = ml_model.LatentDiffusionModel.load_from_checkpoint(
        checkpoint_path=args.checkpoint
    )

    model.eval()
    model.to(device)

    rdi_as = []
    rdi_bs = []

    with torch.no_grad():

        global_sample_id = 0

        for _, batch in enumerate(val_loader):

            x_cond, x_target = batch
            x_cond = x_cond.to(device)
            x_target = x_target.to(device)

            x_generated = model.sample(cond=x_cond)

            for i in range(x_cond.shape[0]):

                cond_points = x_cond[i].detach().cpu().numpy().T
                target_points = x_target[i].detach().cpu().numpy().T
                generated_points = x_generated[i].detach().cpu().numpy().T

                # print("")
                # print("########")
                # print("sample: ", i)
                # print("\t cond_points: ", cond_points.shape)
                # print("\t target_points: ", target_points.shape)
                # print("\t generated_points: ", generated_points.shape)

                fid_cond = latent_fid(cond_points, generated_points)
                fid_target = latent_fid(target_points, generated_points)
                fid_ref = latent_fid(target_points, cond_points)

                rdi_a = 1 - fid_target/fid_ref
                rdi_b = fid_cond/fid_target

                rdi_as.append(rdi_a)
                rdi_bs.append(rdi_b)

                # diff_cond = mse_similarity(cond_points, generated_points)
                # diff_target = mse_similarity(target_points, generated_points)
                # diff_ref = mse_similarity(target_points, cond_points)

                # print("\t diff_cond: ", diff_cond)
                # print("\t diff_target: ", diff_target)
                # print("\t diff_ref: ", diff_ref)

                # tsne_data = np.concatenate(
                #     [
                #         cond_points,
                #         target_points,
                #         generated_points
                #     ],
                #     axis=0
                # )

                # tsne_labels = np.concatenate(
                #     [
                #         np.full(cond_points.shape[0], "Conditioning"),
                #         np.full(target_points.shape[0], "Target"),
                #         np.full(generated_points.shape[0], "Generated"),
                #     ]
                # )

                # tsne_filename = os.path.join(
                #     args.output_dir,
                #     f"sample_{global_sample_id:06d}_tsne.png"
                # )

                # ml_vis.export_tsne(
                #     data=tsne_data,
                #     labels=tsne_labels,
                #     filename=tsne_filename
                # )

                global_sample_id += 1

                # break
            # break

    print("samples: ", len(rdi_as), " -> ", len(rdi_bs))
    print("rdi_as: ", np.mean(rdi_as), " -> ", np.max(rdi_as), " | ", np.min(rdi_as))
    print("rdi_bs: ", np.mean(rdi_bs), " -> ", np.max(rdi_bs), " | ", np.min(rdi_bs))

if __name__ == "__main__":
    main()
