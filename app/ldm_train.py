"""
"""
import os
import argparse

import torch

import lps_ml.utils.device as ml_device
import lps_ml.model as ml_model
import lps_ml.datasets as ml_db
import lps_ml.utils.general as ml_utils
import lps_ml.utils.default as ml_default

def _main():
    """Main function for the dataset info tables."""

    parser = argparse.ArgumentParser(description="Train an LDM on simple_version of iemanja.")
    builder = ml_db.IemanjaBuilder(ldm_exclusive=True)
    builder.add_argparse_args(parser=parser)
    ml_default.add_training_args(parser, default_output_dir="./result/iemanja_classifier")
    ml_model.LatentDiffusionModel.add_args(parser)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    dm = builder.paired_from_argparse_args(args)
    vae_encoder = ml_default.get_vae_encoder(dm)

    device = ml_device.get_available_device()

    model = ml_model.LatentDiffusionModel.from_args(args, dm)
    model = model.to(device)

    trainer, _ = ml_default.trainer_from_args(args, vae_encoder=vae_encoder)
    trainer.fit(model, dm)

if __name__ == "__main__":
    _main()
