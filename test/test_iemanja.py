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

    builder = ml_db.IemanjaBuilder()

    parser = argparse.ArgumentParser(description="Train an MLP classifier on iara.")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    builder.add_argparse_args(parser=parser)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    dm = builder.from_argparse_args(args)

    print(ml_utils.format_header(60,"Dataset description"))
    print(dm.to_compile_df())
    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"Training"))
    print()
    print(dm.to_df())

    # dm.to_df().to_csv("./result/identified.csv")

    model = ml_model.MLP(
        input_shape=dm.get_sample_shape(),
        hidden_channels=[64, 16],
        n_targets=dm.get_n_targets(),
        dropout=0.2,
        lr=1e-5
    )

    checkpoint_cb = lightning_call.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename=f"iara-{{epoch:02d}}-{{val_loss:.3f}}",
    )
    early_stop_cb = lightning_call.EarlyStopping(monitor="val_loss", patience=4, mode="min")

    logger = lightning_log.TensorBoardLogger(
        "logs",
        name="iara"
    )

    trainer = lightning.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)

    train_acc = _evaluate_accuracy(model, dm.train_dataloader())
    val_acc   = _evaluate_accuracy(model, dm.val_dataloader())
    # test_acc  = _evaluate_accuracy(model, dm.test_dataloader())

    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"Results"))
    print(f"Train accuracy:      {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    # print(f"Test accuracy:       {test_acc:.4f}")
    print(ml_utils.format_header(60))

if __name__ == "__main__":
    _main()
