import os
import argparse
import numpy as np
import pandas as pd
import torch

import lps_ml.utils.device as ml_device
import lps_ml.model as ml_model
import lps_ml.datasets as ml_db
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv
import lps_ml.utils.general as ml_utils

import lps_utils.quantities as lps_qty
import lps_sp.acoustical.analysis as lps_analysis
import lps_sp.acoustical.broadband as lps_bb


# =========================
# Métricas básicas
# =========================

def l2_distance(x, y):
    return np.sqrt(np.mean((x - y) ** 2))


def correlation(x, y):
    return np.corrcoef(x.flatten(), y.flatten())[0, 1]


# =========================
# PSD
# =========================

def psd_distance(x, y, fs):
    _, Pxx1 = lps_bb.psd(x, fs=fs, window_size=4096, overlap=0.5)
    _, Pxx2 = lps_bb.psd(y, fs=fs, window_size=4096, overlap=0.5)

    return np.sqrt(np.mean((Pxx1 - Pxx2) ** 2))


# =========================
# LOFAR / MEL
# =========================

def spectral_distance(x, y, fs, analysis_type):

    S1, _, _ = analysis_type.apply(
        data=x,
        fs=fs
    )

    S2, _, _ = analysis_type.apply(
        data=y,
        fs=fs
    )
    return np.sqrt(np.mean((S1 - S2) ** 2))


# =========================
# Avaliação
# =========================
def evaluate(model, loader, vae_encoder, device, fs=16000):

    model.eval()

    results = []

    with torch.no_grad():
        for batch in loader:

            x1, x2 = batch  # condicionante, target

            x1 = x1.to(device)
            x2 = x2.to(device)

            z_gen = model.sample(cond=x1)

            for i in range(x1.shape[0]):

                z_c = x1[i].cpu().numpy()
                z_t = x2[i].cpu().numpy()
                z_g = z_gen[i].cpu().numpy()

                # Decode
                wav_c = vae_encoder.decode(z_c).reshape(-1)
                wav_t = vae_encoder.decode(z_t).reshape(-1)
                wav_g = vae_encoder.decode(z_g).reshape(-1)

                # =========================
                # TEMPO
                # =========================
                d_ct_time = l2_distance(wav_c, wav_t)
                d_gt_time = l2_distance(wav_g, wav_t)
                d_cg_time = l2_distance(wav_c, wav_g)

                corr_ct = correlation(wav_c, wav_t)
                corr_gt = correlation(wav_g, wav_t)

                # =========================
                # PSD
                # =========================
                d_ct_psd = psd_distance(wav_c, wav_t, fs)
                d_gt_psd = psd_distance(wav_g, wav_t, fs)
                d_cg_psd = psd_distance(wav_c, wav_g, fs)

                # =========================
                # LOFAR
                # =========================
                d_ct_lofar = spectral_distance(
                    wav_c, wav_t, fs,
                    lps_analysis.SpectralAnalysis.LOFAR
                )

                d_gt_lofar = spectral_distance(
                    wav_g, wav_t, fs,
                    lps_analysis.SpectralAnalysis.LOFAR
                )

                d_cg_lofar = spectral_distance(
                    wav_c, wav_g, fs,
                    lps_analysis.SpectralAnalysis.LOFAR
                )

                # =========================
                # MEL
                # =========================
                d_ct_mel = spectral_distance(
                    wav_c, wav_t, fs,
                    lps_analysis.SpectralAnalysis.MELGRAM
                )

                d_gt_mel = spectral_distance(
                    wav_g, wav_t, fs,
                    lps_analysis.SpectralAnalysis.MELGRAM
                )

                d_cg_mel = spectral_distance(
                    wav_c, wav_g, fs,
                    lps_analysis.SpectralAnalysis.MELGRAM
                )

                # =========================
                # Scores geométricos
                # =========================
                eps = 1e-8

                S1_time = 1.0 - (d_gt_time / (d_ct_time + eps))
                S2_time = d_ct_time / (d_cg_time + d_gt_time + eps)

                S1_psd = 1.0 - (d_gt_psd / (d_ct_psd + eps))
                S2_psd = d_ct_psd / (d_cg_psd + d_gt_psd + eps)

                S1_lofar = 1.0 - (d_gt_lofar / (d_ct_lofar + eps))
                S2_lofar = d_ct_lofar / (d_cg_lofar + d_gt_lofar + eps)

                S1_mel = 1.0 - (d_gt_mel / (d_ct_mel + eps))
                S2_mel = d_ct_mel / (d_cg_mel + d_gt_mel + eps)

                results.append({
                    # -------- TEMPO --------
                    "d_ct_time": d_ct_time,
                    "d_gt_time": d_gt_time,
                    "d_cg_time": d_cg_time,
                    "corr_ct": corr_ct,
                    "corr_gt": corr_gt,
                    "S1_time": S1_time,
                    "S2_time": S2_time,

                    # -------- PSD --------
                    "d_ct_psd": d_ct_psd,
                    "d_gt_psd": d_gt_psd,
                    "d_cg_psd": d_cg_psd,
                    "S1_psd": S1_psd,
                    "S2_psd": S2_psd,

                    # -------- LOFAR --------
                    "d_ct_lofar": d_ct_lofar,
                    "d_gt_lofar": d_gt_lofar,
                    "d_cg_lofar": d_cg_lofar,
                    "S1_lofar": S1_lofar,
                    "S2_lofar": S2_lofar,

                    # -------- MEL --------
                    "d_ct_mel": d_ct_mel,
                    "d_gt_mel": d_gt_mel,
                    "d_cg_mel": d_cg_mel,
                    "S1_mel": S1_mel,
                    "S2_mel": S2_mel,
                })

    return pd.DataFrame(results)

# =========================
# MAIN
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--vae-model", type=str, default="/data/models/v0_6M.ts")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=str, default="./reconstruction_metrics.csv")

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

    model = ml_model.LatentDiffusionModel.load_from_checkpoint(
        args.model_checkpoint
    ).to(device)

    splits = {
        # "train": dm.train_dataloader(),
        "val": dm.val_dataloader(),
        "test": dm.test_dataloader(),
    }

    all_results = []

    for split_name, loader in splits.items():
        print(f"Evaluating {split_name}...")

        df = evaluate(model, loader, vae_encoder, device)
        df["split"] = split_name

        all_results.append(df)

    final_df = pd.concat(all_results, ignore_index=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final_df.to_csv(args.output, index=False)

    print("\nResumo:")
    print(final_df.describe())


if __name__ == "__main__":
    main()