"""
Evaluate latent-space reconstruction consistency using PSD + t-SNE.
"""

import os
import argparse
import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import scipy.signal as sci_signal

import lps_utils.quantities as lps_qty
import lps_ml.datasets as ml_db
import lps_ml.audio_processors as ml_procs
import lps_ml.core.cv as ml_cv
import lps_ml.visualization.tsne as ml_vis
import lps_ml.utils.general as ml_utils
import lps_ml.utils.device as ml_device
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.acoustical.analysis as lps_analysis
import lps_sp.signal as lps_sig

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./result/psd_tsne")

    parser.add_argument(
        "--fold_role",
        type=str,
        default=None,
        choices=[e.name for e in ml_cv.FoldRole]
    )


    parser.add_argument(
        "--dynamic_selection",
        type=str,
        default=ml_db.DynamicSelection.FIXED_ONLY.name,
        choices=[e.name for e in ml_db.DynamicSelection]
    )

    parser.add_argument(
        "--channel_selection",
        type=str,
        default=ml_db.ChannelSelection.REFERENCE_ONLY.name,
        choices=[e.name for e in ml_db.ChannelSelection]
    )

    args = parser.parse_args()

    psd_dir = os.path.join(args.output_dir, "psd")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(psd_dir, exist_ok=True)

    dynamic_selection = ml_db.DynamicSelection[args.dynamic_selection]
    channel_selection = ml_db.ChannelSelection[args.channel_selection]

    fs = lps_qty.Frequency.khz(16)
    n_samples = int(2**17)
    overlap = int(2**16)

    dm = ml_db.Iemanja(
        file_processor=ml_procs.SampleProcessor(
            n_samples=n_samples,
            overlap=overlap,
            pipelines=[
                ml_procs.ToFloatConverter()
            ]
        ),
        cv=ml_cv.SimpleSplitCV(),
        dynamic_selection=dynamic_selection,
        channel_selection=channel_selection,
        batch_size=args.batch_size
    )

    dm.setup()

    role = args.fold_role
    role_str = role if role is not None else "ALL"
    if role is None:
        loader = dm.all_dataloader()
    else:
        role = ml_cv.FoldRole[role]
        if role == ml_cv.FoldRole.TRAIN:
            loader = dm.train_dataloader()
        elif role == ml_cv.FoldRole.VALIDATION:
            loader = dm.val_dataloader()
        elif role == ml_cv.FoldRole.TEST:
            loader = dm.test_dataloader()

    vae_processor = ml_procs.VAEEncoder(args.model)

    device = ml_device.get_available_device()
    model = vae_processor.model
    model = model.to(device)
    model.eval()

    all_psd = []
    all_mel = []
    all_labels = []

    for x, y in tqdm.tqdm(loader, desc="Processing Samples", ncols=120):

        x = x.to(device).float()

        with torch.inference_mode():
            x_rec = model(x)

        x = x.detach().cpu().numpy()
        x_rec = x_rec.detach().cpu().numpy()

        for i in range(x.shape[0]):

            target = np.squeeze(x[i])
            reconstructed = np.squeeze(x_rec[i])

            # lps_sig.save_convert_wav(
            #     signal = target,
            #     fs = fs,
            #     filename = os.path.join(psd_dir, f"{len(all_psd)}.wav")
            # )
            # lps_sig.save_convert_wav(
            #     signal = reconstructed,
            #     fs = fs,
            #     filename = os.path.join(psd_dir, f"{len(all_psd)}_reconstructed.wav")
            # )
            # lps_bb.plot_psds(
            #     filename = os.path.join(psd_dir, f"{len(all_data)}_psd.png"),
            #     noises=[target, reconstructed],
            #     labels=["Original", "Reconstructed"],
            #     fs=fs,
            #     window_size=4096,
            #     overlap=0.5,
            # )

            _, target_psd = lps_bb.psd(
                signal=target,
                fs=fs,
                window_size=4096,
                overlap=0.5
            )

            _, reconstructed_psd = lps_bb.psd(
                reconstructed,
                fs=fs,
                window_size=4096,
                overlap=0.5
            )

            target_mel = np.mean(
                lps_analysis.SpectralAnalysis.MELGRAM.apply(
                    data=target,
                    fs=fs.get_hz(),
                    params=lps_analysis.Parameters(
                        n_spectral_pts=2048,
                        overlap=0.5,
                        n_mels=256,
                        log_scale=True
                    )
                )[0],
                axis=1
            )

            reconstructed_mel = np.mean(
                lps_analysis.SpectralAnalysis.MELGRAM.apply(
                    data=reconstructed,
                    fs=fs.get_hz(),
                    params=lps_analysis.Parameters(
                        n_spectral_pts=2048,
                        overlap=0.5,
                        n_mels=256,
                        log_scale=True
                    )
                )[0],
                axis=1
            )

            all_psd.append(target_psd)
            all_mel.append(target_mel)
            all_labels.append(f"{y[i]}")

            all_psd.append(reconstructed_psd)
            all_mel.append(reconstructed_mel)
            all_labels.append(f"{y[i]}_reconstructed")

    psd = np.vstack(all_psd)
    mel = np.vstack(all_mel)
    labels = np.array(all_labels)

    ml_vis.export_tsne(
        data=psd,
        labels=labels,
        filename=os.path.join(args.output_dir, f"psd_tsne_{role_str}.png")
    )
    ml_vis.export_tsne(
        data=mel,
        labels=labels,
        filename=os.path.join(args.output_dir, f"mel_tsne_{role_str}.png")
    )

if __name__ == "__main__":
    main()
