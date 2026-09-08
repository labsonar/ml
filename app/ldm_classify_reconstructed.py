"""
Evaluate an LDM style-transfer model over classifiers using reconstructed audio.

The LDM operates in latent space. The generated latent representation is
decoded to audio using a VAE, and the reconstructed audio is then processed
using the same representation/preprocessing used to train the classifier.
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import seaborn as sns

import torch

import lps_utils.quantities as lps_qty
import lps_ml.audio_processors.model_processors as ml_model_procs
import lps_ml.datasets as ml_db
import lps_ml.utils.general as ml_utils
import lps_ml.utils.device as ml_device
import lps_ml.utils.default as ml_default
import lps_ml.model as ml_model
import lps_ml.utils.metrics as ml_metrics


def predict_classifier(
        classifier: torch.nn.Module,
        x: torch.Tensor,
) -> torch.Tensor:
    """Generate class predictions."""

    output = classifier(x)

    if output.ndim == 1:
        pred = (output >= 0.5).long()

    elif output.ndim == 2 and output.shape[1] == 1:
        pred = (output[:, 0] >= 0.5).long()

    else:
        pred = torch.argmax(output, dim=1)

    return pred

def build_classifier_datamodule(args, representation: str):
    """
    Build an Iemanja datamodule configured with the representation used
    during classifier training.
    """
    default_parser = argparse.ArgumentParser(add_help=False)
    builder = ml_db.IemanjaBuilder()
    builder.add_argparse_args(default_parser)

    default_args = default_parser.parse_args([])

    merged = vars(default_args)
    merged.update(vars(args))
    classifier_args = argparse.Namespace(**merged)
    classifier_args.ie_representation = representation

    dm = builder.from_argparse_args(classifier_args)
    return dm

def _main():

    parser = argparse.ArgumentParser(
        description="Evaluate LDM style transfer using reconstructed audio and classifiers.")

    parser.add_argument("--ldm-checkpoint", type=str, required=True,
        help="Path to LDM checkpoint (.ckpt/dir)")

    parser.add_argument("--ship-model", type=str, default=None,
        help="MLP trained to classify ship class (.ckpt/dir)")
    parser.add_argument("--channel-model", type=str, default=None,
        help="MLP trained to classify ship class (.ckpt/dir)")
    parser.add_argument("--shallow-model", type=str, default=None,
        help="MLP trained to classify ship class (.ckpt/dir)")
    parser.add_argument("--output-dir", type=str, default="./result/ldm/classify")
    parser.add_argument("--model-type", type=str, required=True, choices=["mlp", "cnn1d", "cnn2d"],
        help="Model architecture."
    )

    parser.add_argument("--model-representation", type=str, required=True,
        choices=["raw", "spectral"],
        help=(
            "Audio representation expected by the classifier. "
            "'raw' is used by CNN1D and 'spectral' by CNN2D."
        ),
    )

    # Arguments required by the LDM/Iemanja dataset.
    ldm_builder = ml_db.IemanjaBuilder(ldm_exclusive=True)
    ldm_builder.add_argparse_args(parser=parser)

    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.set_float32_matmul_precision("medium")
    ml_utils.set_seed()

    device = ml_device.get_available_device()

    ldm_ckpt = args.ldm_checkpoint

    if os.path.isdir(ldm_ckpt):
        ldm_ckpt = os.path.join(ldm_ckpt, "best.ckpt")

    ldm = ml_model.LDM.load_from_checkpoint(checkpoint_path=ldm_ckpt)
    ldm.to(device)
    ldm.eval()

    channel_pairs = ldm.get_pairs()

    ship_model = ml_utils.load_model(args.ship_model, args.model_type) \
        if args.ship_model is not None else None
    channel_model = ml_utils.load_model(args.channel_model, args.model_type) \
        if args.channel_model is not None else None
    shallow_model = ml_utils.load_model(args.shallow_model, args.model_type) \
        if args.shallow_model is not None else None

    if ship_model is None and channel_model is None and shallow_model is None:
        print("No classifier models provided. Exiting.")
        return

    if ship_model is not None:
        ship_model.to(device)
        ship_model.eval()

    if channel_model is not None:
        channel_model.to(device)
        channel_model.eval()

    if shallow_model is not None:
        shallow_model.to(device)
        shallow_model.eval()

    ldm_builder = ml_db.IemanjaBuilder(ldm_exclusive=True)
    dm = ldm_builder.paired_from_argparse_args(args)
    dm.setup()
    val_loader = dm.val_dataloader()
    vae_encoder = ml_default.get_vae_encoder(dm)

    ship_selector = ml_db.Iemanja.build_column_as_target("CLASS", map_values=True)
    channel_selector = ml_db.Iemanja.build_column_as_target("LOCAL_ID", map_values=True)
    shallow_selector = ml_db.Iemanja.build_column_as_target("SHALLOW_WATER", map_values=True)

    # Setting to targets to the selectors as done in ml_db.Iemanja
    df = dm.to_df().copy()
    ship_selector.apply(df)
    channel_selector.apply(df)
    shallow_selector.apply(df)

    classifier_dm = build_classifier_datamodule(args=args, representation=args.model_representation)
    processor = classifier_dm.file_processor

    ship_pred = []
    channel_pred = []
    shallow_pred = []

    ship_target = []
    channel_target = []
    shallow_target = []

    with torch.no_grad():

        for in_ch, out_ch in channel_pairs:

            for batch, distance, row_id in val_loader:

                x_cond = batch[in_ch]
                x_cond = x_cond.to(device)

                distance = distance.to(device)

                rows = val_loader.dataset.df.iloc[row_id.numpy()]
                fragment_ids = rows[f"id_fragment_{out_ch}"]
                original_df = dm.get_description_by_fragments(fragment_ids)

                z_generated = ldm.sample(
                    cond=x_cond,
                    distance=distance,
                    input_ch=in_ch,
                    output_ch=out_ch,
                )


                audio = vae_encoder.decode(z_generated)
                audio = audio.detach().cpu().numpy()

                ship_batch_targets = ship_selector.apply(original_df)["Target"].to_numpy()
                channel_batch_targets = channel_selector.apply(original_df)["Target"].to_numpy()
                shallow_batch_targets = shallow_selector.apply(original_df)["Target"].to_numpy()

                for i in range(audio.shape[0]):
                    x_reconstructed = processor.process(lps_qty.Frequency.khz(16), audio[i])

                    for processed_sample in x_reconstructed:

                        processed_sample = torch.from_numpy(processed_sample).float()

                        if processed_sample.ndim == 1:
                            processed_sample = processed_sample.unsqueeze(0).unsqueeze(0)

                        elif processed_sample.ndim == 2:
                                processed_sample = processed_sample.unsqueeze(0)

                        else:
                            raise ValueError(f"Unexpected processed sample dimensions: {processed_sample.shape}")

                        processed_sample = processed_sample.to(device)

                        if ship_model is not None:
                            pred = predict_classifier(classifier=ship_model, x=processed_sample)
                            ship_pred.extend(pred.detach().cpu().numpy())
                            ship_target.append(ship_batch_targets[i])

                        if channel_model is not None:
                            pred = predict_classifier(classifier=channel_model, x=processed_sample)
                            channel_pred.extend(pred.detach().cpu().numpy())
                            channel_target.append(channel_batch_targets[i])

                        if shallow_model is not None:
                            pred = predict_classifier(classifier=shallow_model, x=processed_sample)
                            shallow_pred.extend(pred.detach().cpu().numpy())
                            shallow_target.append(shallow_batch_targets[i])

    model_name = os.path.basename(os.path.dirname(os.path.normpath(ldm_ckpt)))

    result = {}

    if ship_model is not None:
        balanced_accuracy = sk_metrics.balanced_accuracy_score(ship_target, ship_pred)
        macro_f1 = sk_metrics.f1_score(ship_target, ship_pred, average="macro")

        result["ship"] = {
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
        }

        ml_metrics.save_confusion_matrix(
                y_true=ship_target,
                y_pred=ship_pred,
                filename=os.path.join(output_dir, f"{model_name}_ship.png"),
                # labels=np.unique(ship_target).tolist(),
                labels=list(range(2)),
                title="Ship Class Confusion Matrix"
            )

    if channel_model is not None:
        balanced_accuracy = sk_metrics.balanced_accuracy_score(channel_target, channel_pred)
        macro_f1 = sk_metrics.f1_score(channel_target, channel_pred, average="macro")

        result["channel"] = {
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
        }

        ml_metrics.save_confusion_matrix(
                y_true=channel_target,
                y_pred=channel_pred,
                filename=os.path.join(output_dir, f"{model_name}_channel.png"),
                # labels=np.unique(channel_target).tolist(),
                labels=list(range(4)),
                title="Channel Confusion Matrix"
            )

    if shallow_model is not None:
        balanced_accuracy = sk_metrics.balanced_accuracy_score(shallow_target, shallow_pred)
        macro_f1 = sk_metrics.f1_score(shallow_target, shallow_pred, average="macro")

        result["shallow"] = {
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": macro_f1,
            }

        ml_metrics.save_confusion_matrix(
                y_true=shallow_target,
                y_pred=shallow_pred,
                filename=os.path.join(output_dir, f"{model_name}_shallow.png"),
                # labels=np.unique(shallow_target).tolist(),
                labels=list(range(2)),
                title="Shallow Confusion Matrix"
            )

    df = pd.DataFrame(result)
    df.to_csv(os.path.join(output_dir, f"{model_name}.csv"), index=False)

    predictions_df = pd.DataFrame({
        "ship_target": ship_target,
        "ship_pred": ship_pred,
        "channel_target": channel_target,
        "channel_pred": channel_pred,
        "shallow_target": shallow_target,
        "shallow_pred": shallow_pred,
    })

    predictions_df.to_csv(os.path.join(output_dir, f"{model_name}_predictions.csv"), index=False)

if __name__ == "__main__":
    _main()
