"""
IARA
"""
import os
import argparse
import matplotlib.pyplot as plt

import torch
import torch.utils.data as torch_data

import lightning
import lightning.pytorch.loggers as lightning_log
import lightning.pytorch.callbacks as lightning_call

import lps_utils.quantities as lps_qty
import lps_sp.acoustical.analysis as lps_analysis

import lps_ml.utils.device as ml_device
import lps_ml.model as ml_model
import lps_ml.core.cv as ml_cv
import lps_ml.audio_processors as ml_procs
import lps_ml.datasets as ml_db
import lps_ml.utils.general as ml_utils
import lps_ml.datasets.iara as ml_iara
import lps_ml.datasets.selection as ml_sel

class LossPlotCallback(lightning.Callback):

    def __init__(self, output_dir: str):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.output_dir = output_dir

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        if "train/loss" in metrics:
            self.train_losses.append(
                metrics["train/loss"].detach().cpu().item()
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        if "val/loss" in metrics:
            self.val_losses.append(
                metrics["val/loss"].detach().cpu().item()
            )

    def on_fit_end(self, trainer, pl_module):

        plt.figure()
        if self.train_losses:
            plt.semilogx(self.train_losses, label="Train Loss")
        if self.val_losses:
            plt.semilogx(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
        plt.close()

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
    parser.add_argument("--data-dir", type=str, default="/data",
                        help="Directory to store iara data.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate.")
    parser.add_argument("--output-dir", type=str, default="./result/iara")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    dm = ml_db.IARA(
            file_processor=ml_procs.SampleProcessor(
                    n_samples=32,
                    overlap=0,
                    fs_out=lps_qty.Frequency.khz(16),
                    pipelines=[
                        ml_procs.ToFloatConverter(),
                        ml_procs.SpectralProcessor(
                            analysis = lps_analysis.SpectralAnalysis.LOFAR,
                            params = lps_analysis.Parameters(
                                n_spectral_pts=1024,
                                overlap=0.5,
                            )
                        )
                    ]
                ),
            cv = ml_cv.SimpleSplitCV(),
            data_collection = ml_iara.DC.OS,
            selection = ml_sel.Selector(
                target = ml_iara.ShipBackgroundClassifier(),
            ),
            batch_size=args.batch_size,
            num_workers=0,
    )
    dm.num_workers = 0

    print(ml_utils.format_header(60,"Dataset description"))
    print(dm.to_compile_df())
    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"Training"))
    print()
    print(dm.to_df())

    dm.setup()
    train_loader = dm.train_dataloader()

    x1, x2 = next(iter(train_loader))

    print("x shape:", x1.shape)
    print("y shape:", x2.shape)

    model = ml_model.CNN2D(
        input_shape=dm.get_sample_shape(),
        conv_n_neurons=[16, 32, 64],
        classification_n_neurons=[128, 32],
        n_targets=dm.get_n_targets(),
        lr=args.lr
    )

    callbacks = [
        lightning_call.ModelCheckpoint(
            dirpath=args.output_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="ldm-{epoch:02d}"
        ),
        lightning_call.EarlyStopping(
            monitor="val/loss",
            min_delta=0.001,
            patience=15,
            verbose=True,
            mode="min"
        ),
        lightning_call.LearningRateMonitor(logging_interval='epoch'),
        LossPlotCallback(args.output_dir),
    ]

    trainer = lightning.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

    train_acc = _evaluate_accuracy(model, dm.train_dataloader())
    val_acc   = _evaluate_accuracy(model, dm.val_dataloader())
    test_acc  = _evaluate_accuracy(model, dm.test_dataloader())

    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"Results"))
    print(f"Train accuracy:      {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")
    print(ml_utils.format_header(60))

if __name__ == "__main__":
    _main()
