#!/usr/bin/env python3
"""
plot_umap_projection.py

Project a dataset and optionally a WAV file into a previously trained UMAP space.
"""

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as scipy_wav
import torch

import lps_ml.datasets as ml_db
import lps_ml.core.cv as ml_cv


def _get_loader(dm, fold_role):

    if fold_role is None:
        return dm.all_dataloader()

    role = ml_cv.FoldRole[fold_role]

    if role == ml_cv.FoldRole.TRAIN:
        return dm.train_dataloader()

    if role == ml_cv.FoldRole.VALIDATION:
        return dm.val_dataloader()

    if role == ml_cv.FoldRole.TEST:
        return dm.test_dataloader()

    raise ValueError(f"Unsupported fold role: {fold_role}")


def _extract_latent_and_labels(loader, latent_mode="flatten"):

    all_data = []
    all_labels = []

    for x, y in loader:

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        if x.ndim > 2:

            if latent_mode == "flatten":
                x = x.reshape(x.shape[0], -1)

            elif latent_mode == "samples":
                B, D, T = x.shape
                x = np.transpose(x, (0, 2, 1))
                x = x.reshape(B * T, D)
                y = np.repeat(y, T)

            else:
                raise ValueError(latent_mode)

        all_data.append(x)
        all_labels.append(y)

    return np.vstack(all_data), np.concatenate(all_labels)


def main():

    builder = ml_db.IemanjaBuilder(vae_exclusive=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("--umap-model", required=True)

    parser.add_argument(
        "--fold_role",
        default=None,
        choices=[f.name for f in ml_cv.FoldRole]
    )

    parser.add_argument(
        "--latent-mode",
        default="flatten",
        choices=["flatten", "samples"]
    )

    parser.add_argument("--wav", default=None)
    parser.add_argument("--output_dir", default="./result/umap_projection")

    builder.add_argparse_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.umap_model, "rb") as f:
        reducer = pickle.load(f)

    dm = builder.from_argparse_args(args)
    dm.batch_size = 1
    dm.num_workers = 0
    dm.setup()

    loader = _get_loader(dm, args.fold_role)

    embedding_wav = None

    if args.wav is not None:

        fs, signal = scipy_wav.read(args.wav)

        if signal.ndim != 1:
            signal = signal[:,0]

        processed = dm.file_processor.process(fs=fs, data=signal)
        processed = np.array(processed)
        print("processed: ", processed.shape)

        if processed.ndim > 2:

            if args.latent_mode == "flatten":

                processed = processed.reshape(processed.shape[0], -1)

            elif args.latent_mode == "samples":

                B, D, T = processed.shape

                processed = np.transpose(processed, (0, 2, 1))
                processed = processed.reshape(B * T, D)

            else:
                raise ValueError(
                    f"Unknown latent_mode: {args.latent_mode}"
                )

        embedding_wav = reducer.transform(processed)

    plt.figure(figsize=(10, 10))


    print("Extracting dataset latents...")
    data, labels = _extract_latent_and_labels(
        loader,
        latent_mode=args.latent_mode
    )

    print("Projecting dataset...")
    embedding_dataset = reducer.transform(data)

    for label in np.unique(labels):

        mask = labels == label

        plt.scatter(
            embedding_dataset[mask, 0],
            embedding_dataset[mask, 1],
            s=6,
            alpha=0.5,
            label=str(label)
        )

    if embedding_wav is not None:

        # plt.plot(
        #     embedding_wav[:, 0],
        #     embedding_wav[:, 1],
        #     linewidth=2,
        #     label="wav trajectory",
        #     color="black"
        # )

        plt.scatter(
            embedding_wav[:, 0],
            embedding_wav[:, 1],
            s=10,
            color="black",
            label="wav samples",
        )

    plt.legend()
    plt.tight_layout()

    fig_file = os.path.join(
        args.output_dir,
        "umap_projection.png"
    )

    plt.savefig(fig_file, dpi=300)
    plt.close()

    print(f"Saved: {fig_file}")


if __name__ == "__main__":
    main()