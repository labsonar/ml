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
import shutil

import pandas as pd
import sklearn.metrics as sk_metrics

import torch
import torch.utils.data as torch_data

import lightning
import lightning.pytorch.loggers as lightning_log
import lightning.pytorch.callbacks as lightning_call

import lps_ml.model as ml_model
import lps_ml.datasets as ml_db
import lps_ml.utils.device as ml_device
import lps_ml.utils.general as ml_utils


class DomainDataset(torch_data.Dataset):
    """
    Wrap an existing dataset and replace its target with a
    domain label.

    domain_label = 0 -> dataset 1
    domain_label = 1 -> dataset 2
    """

    def __init__(
            self,
            dataset: torch_data.Dataset,
            domain_label: int,
    ):
        self.dataset = dataset
        self.domain_label = domain_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        y = torch.tensor(self.domain_label, dtype=torch.long)

        return x, y


def build_combined_loader(
        loader1: torch_data.DataLoader,
        loader2: torch_data.DataLoader,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
):
    """
    Combine two Iemanja datasets into a binary domain dataset.
    """

    dataset1 = DomainDataset(loader1.dataset, domain_label=0)
    dataset2 = DomainDataset(loader2.dataset, domain_label=1)

    combined_dataset = torch_data.ConcatDataset([dataset1, dataset2])

    return torch_data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


def evaluate(model: torch.nn.Module, dataloader: torch_data.DataLoader):
    """
    Evaluate binary domain classification.
    """

    device = ml_device.get_available_device()

    model.eval()
    model.to(device)

    y_true = []
    y_pred = []

    with torch.inference_mode():

        for x, y in dataloader:

            if isinstance(x, list):
                raise RuntimeError(
                    "Domain classifier expects a non-paired DataLoader."
                )

            x = x.to(device)
            y = y.to(device)

            output = model(x)

            if output.ndim == 1:

                pred = (output >= 0.5).long()

            elif output.ndim == 2 and output.shape[1] == 1:

                pred = (output[:, 0] >= 0.5).long()

            else:

                pred = torch.argmax(output, dim=1)

            y_true.extend(
                y.cpu().numpy()
            )

            y_pred.extend(
                pred.cpu().numpy()
            )

    balanced_accuracy = sk_metrics.balanced_accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = sk_metrics.f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    return balanced_accuracy, macro_f1


def build_model(args, dm):

    input_shape = dm.get_sample_shape()

    print()
    print("Model configuration")
    print("-------------------")
    print(f"Architecture : {args.model}")
    print(f"Input shape  : {input_shape}")
    print("Targets      : 2")

    if args.model == "cnn1d":

        model = ml_model.CNN1D(
            input_shape=input_shape,

            # Feature extractor
            conv_n_neurons=[4, 8, 16, 32],
            conv_activation=torch.nn.LeakyReLU,
            conv_pooling=torch.nn.MaxPool1d,
            conv_pooling_size=[4, 4, 4, 4],
            conv_dropout=0.4,
            batch_norm=torch.nn.BatchNorm1d,
            kernel_size=5,

            # Classification head
            classification_n_neurons=64,
            n_targets=2,
            classification_dropout=0.4,
            classification_norm=None,
            classification_output_activation=torch.nn.Sigmoid,

            # Optimization
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    elif args.model == "cnn2d":

        model = ml_model.CNN2D(
            input_shape=input_shape,

            # Feature extractor
            conv_n_neurons=[16, 32, 64],
            conv_activation=torch.nn.ReLU,
            conv_pooling=None,
            conv_pooling_size=[4, 2],
            conv_dropout=0.5,
            batch_norm=torch.nn.BatchNorm2d,
            kernel_size=5,

            # Classification head
            classification_n_neurons=32,
            n_targets=2,
            classification_dropout=0.5,
            classification_norm=None,
            classification_hidden_activation=torch.nn.ReLU,
            classification_output_activation=torch.nn.Sigmoid,

            # Optimization
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    else:
        raise ValueError(
            f"Unknown model: {args.model}"
        )

    return model


def build_datamodule(builder, args, dataset_dir):
    """
    Build one Iemanja DataModule.
    """
    args.ie_dataset_dir = dataset_dir
    dm = builder.from_argparse_args(args)
    dm.setup()
    return dm


def verify_sample_shapes(dm1, dm2):

    shape1 = dm1.get_sample_shape()
    shape2 = dm2.get_sample_shape()

    print()
    print("Input shapes")
    print("------------")
    print(f"Dataset 1: {shape1}")
    print(f"Dataset 2: {shape2}")

    if shape1 != shape2:
        raise RuntimeError(
            "Dataset 1 and dataset 2 have different sample shapes:\n"
            f"dataset1 = {shape1}\n"
            f"dataset2 = {shape2}"
        )


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

    parser.add_argument("--model", choices=["cnn1d", "cnn2d"], default="cnn2d",
        help="CNN architecture.")

    parser.add_argument("--max-epochs", type=int, default=2000,
        help="Maximum number of training epochs.")

    parser.add_argument("--lr", type=float, default=1e-3,
        help="Learning rate.")

    parser.add_argument("--weight-decay", type=float, default=1e-4,
        help="Weight decay.")

    parser.add_argument("--output-dir", type=str, default="./result/original_vs_vae",
        help="Output directory.")

    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001,
        help="Minimum validation loss improvement.")

    parser.add_argument("--early-stopping-patience", type=int, default=200,
        help="Early stopping patience.")

    builder = ml_db.IemanjaBuilder()
    builder.add_argparse_args(parser)

    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    ml_utils.set_seed()

    os.makedirs(args.output_dir, exist_ok=True)

    log_dir = os.path.join(args.output_dir,"log")
    model_best = os.path.join(args.output_dir,"best.ckpt")
    model_last = os.path.join(args.output_dir,"last.ckpt")

    print()
    print("=" * 70)
    print("Building datasets")
    print("=" * 70)

    dm1 = build_datamodule(builder=builder, args=args, dataset_dir=args.dataset1_dir)
    dm2 = build_datamodule(builder=builder, args=args, dataset_dir=args.dataset2_dir)

    verify_sample_shapes(dm1, dm2)

    loader1_train = dm1.train_dataloader(shuffle=False)
    loader1_val = dm1.val_dataloader(shuffle=False)
    loader1_test = dm1.test_dataloader(shuffle=False)

    loader2_train = dm2.train_dataloader(shuffle=False)
    loader2_val = dm2.val_dataloader(shuffle=False)
    loader2_test = dm2.test_dataloader(shuffle=False)

    print()
    print("=" * 70)
    print("Building combined domain datasets")
    print("=" * 70)

    train_loader = build_combined_loader(
        loader1=loader1_train,
        loader2=loader2_train,
        batch_size=args.ie_batch_size,
        shuffle=True,
        num_workers=args.ie_num_workers,
    )

    val_loader = build_combined_loader(
        loader1=loader1_val,
        loader2=loader2_val,
        batch_size=args.ie_batch_size,
        shuffle=False,
        num_workers=args.ie_num_workers,
    )

    test_loader = build_combined_loader(
        loader1=loader1_test,
        loader2=loader2_test,
        batch_size=args.ie_batch_size,
        shuffle=False,
        num_workers=args.ie_num_workers,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples  : {len(val_loader.dataset)}")
    print(f"Test samples : {len(test_loader.dataset)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = build_model(args=args, dm=dm1)

    # ------------------------------------------------------------------
    # Callbacks / logger
    # ------------------------------------------------------------------

    checkpoint_cb = lightning_call.ModelCheckpoint(
        dirpath=log_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
    )

    callbacks = [
        checkpoint_cb,

        lightning_call.EarlyStopping(
            monitor="val/loss",
            min_delta=args.early_stopping_min_delta,
            patience=args.early_stopping_patience,
            verbose=True,
            mode="min",
        ),

        lightning_call.LearningRateMonitor(
            logging_interval="epoch",
        ),
    ]

    logger = lightning_log.TensorBoardLogger(
        log_dir,
        name="domain_classifier",
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    trainer = lightning.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    if checkpoint_cb.last_model_path:
        shutil.copy2(checkpoint_cb.last_model_path, model_last)

    if checkpoint_cb.best_model_path:
        shutil.copy2(checkpoint_cb.best_model_path, model_best)

    if args.model == "cnn1d":
        best_model = ml_model.CNN1D.load_from_checkpoint(model_best)
    else:
        best_model = ml_model.CNN2D.load_from_checkpoint(model_best)

    results = []

    for split, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:

        balanced_accuracy, macro_f1 = evaluate(
            model=best_model,
            dataloader=loader,
        )

        results.append(
            {
                "dataset": split,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
            }
        )

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)

    print()
    print("=" * 70)
    print("DOMAIN CLASSIFICATION RESULTS")
    print("=" * 70)

    print(metrics_df.to_string(index=False))

    print()
    print("=" * 70)
    print("Output")
    print("=" * 70)

    print(f"Metrics: {args.output_dir}/metrics.csv")
    print(f"Best model: {model_best}")
    print("=" * 70)


if __name__ == "__main__":
    _main()
