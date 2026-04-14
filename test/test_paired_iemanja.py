"""
Olocum
"""
import argparse

import torch
import torch.utils.data as torch_data

import lightning
import lightning.pytorch.loggers as lightning_log
import lightning.pytorch.callbacks as lightning_call

import lps_utils.quantities as lps_qty

import lps_ml.utils.device as ml_device
import lps_ml.model as ml_model
import lps_ml.core.cv as ml_cv
import lps_ml.audio_processors as ml_procs
import lps_ml.datasets as ml_db
import lps_ml.utils.general as ml_utils
import lps_ml.datasets.synthetic as ml_synthetic

def _evaluate_accuracy(model: torch.nn.Module,
                      dataloader: torch_data.DataLoader):
    device = ml_device.get_available_device()
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            if out.ndim == 1:
                preds = (out > 0.5).long()
            else:
                preds = torch.argmax(out, dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    return acc

def _main():
    """Main function for the dataset info tables."""

    parser = argparse.ArgumentParser(description="Train an MLP classifier on iara.")
    parser.add_argument("--model", type=str, default="/data/models/v0_4M6.ts")
    parser.add_argument("--data-dir", type=str, default="/data",
                        help="Directory to store iara data.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    n_samples=int(2**17)    #8.192s
    overlap=int(2**16)      #4.096s
    latent_compactness = int(2**10)

    dm = ml_db.IemanjaPaired(
            file_processor=ml_procs.SampleProcessor(
                    n_samples=int(n_samples/latent_compactness),
                    overlap=int(overlap/latent_compactness),
                    pipelines=[
                        ml_procs.ToFloatConverter(),
                        ml_procs.VAEEncoder(args.model)
                    ]
                ),
            cv = ml_cv.SimpleSplitCV(),
            simple_version=True,
            batch_size=16
            )
    dm.setup()

    print(ml_utils.format_header(60,"Dataset description"))
    print(dm.to_compile_df())
    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"pairs_to_full_df"))
    print(dm.pairs_to_df())
    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"Training"))
    print()
    print(dm.to_df())

    dm.to_df().to_csv("./result/paired.csv")
    dm.pairs_to_df().to_csv("./result/pairs_to_full_df.csv")

    train_loader = dm.train_dataloader()

    x1, x2 = next(iter(train_loader))

    print("x1 shape:", x1.shape)
    print("x2 shape:", x2.shape)

if __name__ == "__main__":
    _main()
