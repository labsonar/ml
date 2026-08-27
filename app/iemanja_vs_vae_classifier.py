#!/usr/bin/env python3
"""
Train a binary classifier to distinguish two Iemanja domains.

Domain 0 = dataset 1
Domain 1 = dataset 2

For each split:

    dataset1/train + dataset2/train -> combined train
    dataset1/val   + dataset2/val   -> combined val
    dataset1/test  + dataset2/test  -> combined test

The classifier therefore learns to distinguish the two domains,
rather than the original Iemanja target classes.
"""

import os
import argparse

import torch

import lps_ml.core.datamodule as ml_dm
import lps_ml.model as ml_model
import lps_ml.datasets as ml_db
import lps_ml.utils.general as ml_utils
import lps_ml.utils.default as ml_default
import lps_ml.utils.metrics as ml_metrics

def _build_datamodule(builder, args, dataset_dir):
    args.ie_dataset_dir = dataset_dir
    dm = builder.from_argparse_args(args)
    dm.setup()
    return dm

def _main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a binary CNN to distinguish two Iemanja domains."
        )
    )

    parser.add_argument("--dataset1-dir", type=str, required=True,
        help="First Iemanja dataset. Domain label = 0.")
    parser.add_argument("--dataset2-dir", type=str, required=True,
        help="Second Iemanja dataset. Domain label = 1.")

    builder = ml_db.IemanjaBuilder()
    builder.add_argparse_args(parser)
    ml_default.add_model_args(parser, models=["mlp", "cnn1d", "cnn2d"])
    ml_default.add_training_args(parser, default_output_dir="./result/iemanja_vs_vae")

    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    ml_utils.set_seed()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 70)
    print("Building datasets")
    print("=" * 70)


    dm1 = _build_datamodule(builder=builder, args=args, dataset_dir=args.dataset1_dir)
    dm2 = _build_datamodule(builder=builder, args=args, dataset_dir=args.dataset2_dir)

    dm1.verify_sample_shapes(dm2)

    loader1_train = dm1.train_dataloader()
    loader1_val = dm1.val_dataloader()
    loader1_test = dm1.test_dataloader()

    loader2_train = dm2.train_dataloader()
    loader2_val = dm2.val_dataloader()
    loader2_test = dm2.test_dataloader()

    print()
    print("=" * 70)
    print("Building combined domain datasets")
    print("=" * 70)

    train_loader = ml_dm.DomainDataset.build_domain_dataloader(
        loader1=loader1_train,
        loader2=loader2_train,
        batch_size=args.ie_batch_size,
        shuffle=True,
        num_workers=args.ie_num_workers,
    )

    val_loader = ml_dm.DomainDataset.build_domain_dataloader(
        loader1=loader1_val,
        loader2=loader2_val,
        batch_size=args.ie_batch_size,
        shuffle=False,
        num_workers=args.ie_num_workers,
    )

    test_loader = ml_dm.DomainDataset.build_domain_dataloader(
        loader1=loader1_test,
        loader2=loader2_test,
        batch_size=args.ie_batch_size,
        shuffle=False,
        num_workers=args.ie_num_workers,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples  : {len(val_loader.dataset)}")
    print(f"Test samples : {len(test_loader.dataset)}")

    model = ml_default.model_from_args(args, dm1)
    trainer, ckpt = ml_default.trainer_from_args(args)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    if args.model == "mlp":
        best_model = ml_model.MLP.load_from_checkpoint(ckpt.get_best())
    elif args.model == "cnn1d":
        best_model = ml_model.CNN1D.load_from_checkpoint(ckpt.get_best())
    else:
        best_model = ml_model.CNN2D.load_from_checkpoint(ckpt.get_best())


    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    with ml_utils.evaluating(best_model):
        metrics_df = ml_metrics.evaluate_splits(best_model, loaders)

    metrics_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path)


    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"Results"))
    print(metrics_df)
    print(ml_utils.format_header(60))


if __name__ == "__main__":
    _main()
