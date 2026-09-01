import os
import typing
import argparse
import numpy as np

import torch
import lightning
import scipy.io.wavfile as scipy_wav

import lps_ml.core.datamodule as ml_core
import lps_ml.model.mlp as lps_mlp
import lps_ml.model.cnn as lps_cnn
import lps_ml.utils.lightning as lps_light
import lps_ml.audio_processors.model_processors as ml_model_procs


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

def trainer_from_args(
        args: argparse.Namespace,
        vae_encoder = None
) -> typing.Tuple[lightning.Trainer, lps_light.ExportableModelCheckpoint]:
    """
    Create a Lightning Trainer from common training arguments.
    """
    return lps_light.default_trainer(
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.val_every_n_epoch,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        vae_encoder=vae_encoder
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


def add_ldm_eval_args(parser: argparse.ArgumentParser) -> None:
    """
    Add the --ldm-checkpoint arguments
    """
    parser.add_argument("--ldm-checkpoint", type=str, required=True,
        help="Path to a trained LatentDiffusionModel checkpoint (.ckpt).")

def parse_model_specs(specs: typing.List[str], default_compactness: int) -> typing.Dict[str, int]:
    """
    Parse a list of "model_path" or "model_path:compactness"
    """
    parsed = {}

    for spec in specs:
        if ":" in spec:
            path, comp = spec.rsplit(":", 1)
            try:
                compactness = int(comp)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid compactness value in '{spec}'. "
                    "Use format model_path:latent_compactness"
                ) from exc
        else:
            path = spec
            compactness = default_compactness

        parsed[os.path.abspath(path)] = compactness

    return parsed


def process_wav_to_latent(
    dm,
    wav_path: str,
    latent_mode: str = "flatten",
) -> np.ndarray:
    """
    Run a single WAV file through a DataModule's file_processor and reshape the result according to
    `latent_mode`
    """

    fs, signal = scipy_wav.read(wav_path)

    if signal.ndim != 1:
        signal = signal[:, 0]

    processed = dm.file_processor.process(fs=fs, data=signal)
    processed = np.array(processed)

    if processed.ndim > 2:
        if latent_mode == "flatten":
            processed = processed.reshape(processed.shape[0], -1)
        elif latent_mode == "samples":
            b, d, t = processed.shape
            processed = np.transpose(processed, (0, 2, 1))
            processed = processed.reshape(b * t, d)
        else:
            raise ValueError(f"Unknown latent_mode: {latent_mode}")

    return processed

def find_pipeline(processor, pipeline_cls: typing.Type):
    """
    Find the first pipeline of a given type inside a processor's
    `.pipelines` list (e.g. SampleProcessor/TimeProcessor).
    """
    for pipeline in getattr(processor, "audio_pipelines", []):
        if isinstance(pipeline, pipeline_cls):
            return pipeline

    raise ValueError(f"No pipeline of type {pipeline_cls.__name__} found in {processor}.")

def get_vae_encoder(dm):
    """
    Convenience wrapper: find the VAEEncoder pipeline inside a DataModule's
    """
    return find_pipeline(dm.file_processor, ml_model_procs.VAEEncoder)