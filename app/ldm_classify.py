"""
Evaluate a trained Latent Diffusion Model (LDM) over classifiers.
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import seaborn as sns

import torch

import lps_ml.datasets as ml_db
import lps_ml.utils.general as ml_utils
import lps_ml.utils.device as ml_device
import lps_ml.model as ml_model

def predict_classifier(classifier: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """ Generate class predictions from latent representations. """

    output = classifier(x)

    if output.ndim == 1:
        pred = (output >= 0.5).long()

    elif output.ndim == 2 and output.shape[1] == 1:
        pred = (output[:, 0] >= 0.5).long()

    else:
        pred = torch.argmax(output, dim=1)

    return pred

def save_confusion_matrix(
        y_true,
        y_pred,
        labels,
        filename,
        title,
):
    """
    Save a confusion matrix using seaborn.
    """

    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=labels,)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def _main():

    parser = argparse.ArgumentParser(
        description="Evaluate LDM latent reconstructions."
    )
    parser.add_argument("--ldm-checkpoint", type=str, required=True,
        help="Path to LDM checkpoint (.ckpt/dir)")
    parser.add_argument("--ship-model", type=str, default=None,
        help="MLP trained to classify ship class (.ckpt/dir)")
    parser.add_argument("--channel-model", type=str, default=None,
        help="MLP trained to classify ship class (.ckpt/dir)")
    parser.add_argument("--shallow-model", type=str, default=None,
        help="MLP trained to classify ship class (.ckpt/dir)")
    parser.add_argument("--output-dir", type=str, default="./result/ldm/classify")

    builder = ml_db.IemanjaBuilder(ldm_exclusive=True)
    builder.add_argparse_args(parser=parser)

    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.set_float32_matmul_precision("medium")
    ml_utils.set_seed()

    device = ml_device.get_available_device()

    dm = builder.paired_from_argparse_args(args)
    dm.setup()
    val_loader = dm.val_dataloader()

    ckpt = args.ldm_checkpoint
    if os.path.isdir(ckpt):
        ckpt = os.path.join(ckpt, "best.ckpt")
    model = ml_model.LDM.load_from_checkpoint(checkpoint_path=ckpt)
    model.to(device)
    model.eval()
    channel_pairs = model.get_pairs()

    ship_model = ml_utils.load_model(args.ship_model, "mlp") \
        if args.ship_model is not None else None
    channel_model = ml_utils.load_model(args.channel_model, "mlp") \
        if args.channel_model is not None else None
    shallow_model = ml_utils.load_model(args.shallow_model, "mlp") \
        if args.shallow_model is not None else None

    if ship_model is None and channel_model is None and shallow_model is None:
        print("No classifier models provided. Exiting.")
        return

    ship_selector = ml_db.Iemanja.build_column_as_target("CLASS", map_values=True)
    channel_selector = ml_db.Iemanja.build_column_as_target("LOCAL_ID", map_values=True)
    shallow_selector = ml_db.Iemanja.build_column_as_target("SHALLOW_WATER", map_values=True)

    # Setting to targets to the selectors as done in ml_db.Iemanja
    df = dm.to_df().copy()
    ship_selector.apply(df)
    channel_selector.apply(df)
    shallow_selector.apply(df)


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
                x_target = batch[out_ch]

                x_cond = x_cond.to(device)
                x_target = x_target.to(device)
                distance = distance.to(device)

                rows = val_loader.dataset.df.iloc[row_id.numpy()]
                fragment_ids = rows[f"id_fragment_{out_ch}"]
                original_df = dm.get_description_by_fragments(fragment_ids)

                x_generated = model.sample(cond=x_cond,
                                           distance=distance,
                                           input_ch=in_ch,
                                           output_ch=out_ch)

                if ship_model is not None:
                    pred = predict_classifier(classifier=ship_model, x=x_generated)
                    ship_pred.extend(pred.detach().cpu().numpy())
                    ship_target.extend(ship_selector.apply(original_df)["Target"].to_numpy())

                if channel_model is not None:
                    pred = predict_classifier(classifier=channel_model, x=x_generated)
                    channel_pred.extend(pred.detach().cpu().numpy())
                    channel_target.extend(channel_selector.apply(original_df)["Target"].to_numpy())

                if shallow_model is not None:
                    pred = predict_classifier(classifier=shallow_model, x=x_generated)
                    shallow_pred.extend(pred.detach().cpu().numpy())
                    shallow_target.extend(shallow_selector.apply(original_df)["Target"].to_numpy())


    model_name = os.path.basename(os.path.dirname(os.path.normpath(ckpt)))

    result = {}

    if ship_model is not None:
        balanced_accuracy = sk_metrics.balanced_accuracy_score(ship_target, ship_pred)
        macro_f1 = sk_metrics.f1_score(ship_target, ship_pred, average="macro")

        result["ship"] = {
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
        }

        save_confusion_matrix(
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

        save_confusion_matrix(
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

        save_confusion_matrix(
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
