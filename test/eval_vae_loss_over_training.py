"""
Evaluate STFT and Mel reconstruction losses of VAE checkpoints along training,
for different model variants (model_ids) and different training steps.
"""
import os
import argparse
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

import lps_ml.utils.device as ml_device
import lps_ml.datasets as ml_db
import lps_ml.utils.general as ml_utils
import lps_ml.utils.sonar_loss as ml_loss


def evaluate_model(model_path: str,
                    val_loader,
                    sonar_loss: ml_loss.SonarLoss,
                    device: torch.device) -> dict:
    """
    Loads a single VAE (TorchScript) model and evaluates its reconstruction
    quality (STFT and Mel losses) over a validation DataLoader.

    Args:
        model_path: path to the .ts checkpoint.
        val_loader: validation DataLoader yielding raw waveforms (x, y).
        sonar_loss: SonarLoss instance (used only for its stft_loss/mel_loss).
        device: torch device to run inference on.

    Returns:
        dict with keys "stft" and "mel" (mean loss over the validation set).
    """

    model = torch.jit.load(model_path)
    model.to(device)
    model.eval()

    stft_values = []
    mel_values = []

    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)

            with torch.inference_mode():
                recon = model(x)

            if recon.shape != x.shape:
                min_len = min(recon.shape[-1], x.shape[-1])
                recon = recon[..., :min_len]
                x_cmp = x[..., :min_len]
            else:
                x_cmp = x

            stft_value = sonar_loss.stft_loss(x_cmp, recon)
            mel_value = sonar_loss.mel_loss(x_cmp, recon)

            stft_values.append(stft_value.item())
            mel_values.append(mel_value.item())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "stft": float(np.mean(stft_values)),
        "mel": float(np.mean(mel_values)),
    }


def plot_metric(results: dict,
                 model_ids: list,
                 steps: list,
                 metric: str,
                 output_path: str):
    """ Plot a loss metric vs training steps, with one line per model. """

    plt.figure(figsize=(8, 6))

    for model_id in model_ids:
        y = [results.get(model_id, {}).get(step, {}).get(metric, np.nan) for step in steps]
        plt.plot(steps, y, marker="o", label=model_id)

    plt.xlabel("Training step")
    plt.ylabel(f"{metric.upper()} loss")
    plt.title(f"{metric.upper()} reconstruction loss over training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def _main():

    builder = ml_db.IemanjaBuilder(time_exclusive=True)

    parser = argparse.ArgumentParser(
        description="Evaluate STFT/Mel reconstruction loss of VAE checkpoints "
                    "over training steps, for several model variants."
    )

    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing the .ts checkpoints, named as "
                             "{model_id}_ep{step}.ts")
    parser.add_argument("--output-dir", type=str, default="./result/vae_loss_over_training",
                        help="Directory to save the resulting plots and CSV summary.")

    builder.add_argparse_args(parser=parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ml_utils.set_seed()
    device = ml_device.get_available_device()

    model_ids = ["ch5", "ch5_s1", "default"]
    steps = [200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

    # Dataset built with the existing structures, raw waveforms only
    # (time_exclusive=True skips any VAE/spectral step in the file_processor).
    dm = builder.from_argparse_args(args)
    dm.setup()

    print(ml_utils.format_header(60, "Dataset description"))
    print(dm.to_compile_df())
    print(ml_utils.format_header(60))

    val_loader = dm.val_dataloader()

    sonar_loss = ml_loss.SonarLoss(
        stft_factor=1.0,
        mel_factor=1.0,
        lofar_factor=0.0,
        demon_factor=0.0,
    )

    results = collections.defaultdict(dict)
    rows = []

    for model_id in model_ids:
        for step in steps:

            model_path = os.path.join(args.model_dir, f"{model_id}_ep{step}.ts")

            if not os.path.exists(model_path):
                print(f"[WARNING] Checkpoint not found, skipping: {model_path}")
                continue

            print(ml_utils.format_header(60, f"{model_id} @ step {step}"))

            try:
                metrics = evaluate_model(
                    model_path=model_path,
                    val_loader=val_loader,
                    sonar_loss=sonar_loss,
                    device=device,
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"[ERROR] Failed to evaluate {model_path}: {e}")
                continue

            print(f"\tstft: {metrics['stft']:.6f}   mel: {metrics['mel']:.6f}")

            results[model_id][step] = metrics
            rows.append({
                "model_id": model_id,
                "step": step,
                "stft": metrics["stft"],
                "mel": metrics["mel"],
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "loss_over_training.csv")
    df.to_csv(csv_path, index=False)

    print(ml_utils.format_header(60, "Results"))
    print(df)
    print(f"\nSaved metrics table: {csv_path}")

    for metric in ["stft", "mel"]:
        output_path = os.path.join(args.output_dir, f"{metric}_loss_over_training.png")
        plot_metric(
            results=results,
            model_ids=model_ids,
            steps=steps,
            metric=metric,
            output_path=output_path,
        )
        print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    _main()