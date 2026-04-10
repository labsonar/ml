import os
import random
import argparse
import torch
import torchvision
import matplotlib.pyplot as plt

import lightning
import lightning.pytorch.callbacks as lightning_call

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_sig
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.acoustical.analysis as lps_analysis
import lps_ml.utils.general as ml_utils
import lps_ml.datasets as ml_db
import lps_ml.model as ml_model
import lps_ml.core.cv as ml_cv
import lps_ml.audio_processors as ml_procs
import lps_ml.datasets.selection as ml_sel
import lps_ml.model.audio_vae as lps_audio_vae


def _main():
    parser = argparse.ArgumentParser(
        description="Test AudioFolder dataset"
    )
    parser.add_argument("--model", type=str, default="/data/models/v0_4M6.ts")
    parser.add_argument("input_dir", type=str, help="Root directory containing class subfolders")
    args = parser.parse_args()

    fs=lps_qty.Frequency.khz(16)
    n_samples=int(2**7)
    overlap=int(2**6)

    dm = ml_db.AudioFolder(
        file_processor=ml_procs.SampleProcessor(
                fs_out=fs,
                n_samples=n_samples,
                overlap=overlap,
                pipelines=[
                    ml_procs.ToFloatConverter(),
                    ml_procs.VAEEncoder(args.model)
                ]
            ),
        cv=ml_cv.SimpleSplitCV(),
        input_dir=args.input_dir,
        batch_size=16,
        num_workers=1
    )

    dm.setup()
    dl = dm.train_dataloader()

    batch = next(iter(dl))

    print(type(batch))

    if isinstance(batch, torch.Tensor):
        print("Shape:", batch.shape)

    elif isinstance(batch, (list, tuple)):
        print("Len batch:", len(batch))
        print("Shape[0]:", batch[0].shape)

    elif isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.shape}")

if __name__ == "__main__":
    _main()
