import typing
import argparse

import torch
import lightning

import lps_ml.core.datamodule as ml_core
import lps_ml.model.mlp as lps_mlp
import lps_ml.model.cnn as lps_cnn
import lps_ml.utils.lightning as lps_light

def add_training_args(
    parser: argparse.ArgumentParser,
    default_output_dir: str,
    default_max_epochs: int = 2000,
    default_early_stopping_patience: int = 200,
    default_early_stopping_min_delta: float = 0.001,
) -> argparse._ArgumentGroup:
    """
    Add the common set of training-related arguments to a parser.
    """
    group = parser.add_argument_group("Training", "Common training options")

    group.add_argument("--max-epochs", type=int, default=default_max_epochs,
        help="Maximum number of training epochs.")

    group.add_argument("--output-dir", type=str, default=default_output_dir,
        help="Output directory for checkpoints, logs and metrics.")

    group.add_argument("--early-stopping-patience", type=int,
        default=default_early_stopping_patience,
        help="Number of validation epochs without improvement before stopping.")

    group.add_argument("--early-stopping-min-delta", type=float,
        default=default_early_stopping_min_delta,
        help="Minimum validation loss improvement required to reset patience.")

    group.add_argument("--val-every-n-epoch", type=int, default=1,
        help="Number of epochs between validation checks.")

    return group

def trainer_from_args(args: argparse.Namespace) -> \
        typing.Tuple[lightning.Trainer, lps_light.ExportableModelCheckpoint]:
    """
    Create a Lightning Trainer from common training arguments.
    """
    return lps_light.default_trainer(
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.val_every_n_epoch,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )

def add_model_args(parser: argparse.ArgumentParser, models: typing.List[str] | None = None) -> None:
    """Add command-line arguments for the requested models."""

    if models is None:
        models = ["mlp", "cnn1d", "cnn2d"]

    parser.add_argument("--model", type=str, required=True, choices=models,
        help="Model architecture."
    )

    if "mlp" in models:
        lps_mlp.MLP.add_args(parser)

    if "cnn1d" in models:
        lps_cnn.CNN1D.add_args(parser)

    if "cnn2d" in models:
        lps_cnn.CNN2D.add_args(parser)

def model_from_args(args: argparse.Namespace, dm: ml_core.BaseDataModule) -> \
    torch.nn.Module:
    """Create the selected model from command-line arguments."""

    if args.model == "mlp":
        return lps_mlp.MLP.from_args(args, dm)

    if args.model == "cnn1d":
        return lps_cnn.CNN1D.from_args(args, dm)

    if args.model == "cnn2d":
        return lps_cnn.CNN2D.from_args(args, dm)

    raise ValueError(f"Unknown model: {args.model}")
