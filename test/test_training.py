"""
Train an MLP on the MNIST dataset using PyTorch Lightning.

tensorboard --logdir logs --host 0.0.0.0 --port 7088
"""
import argparse

import torch
import torch.utils.data as torch_data

import lightning
import lightning.pytorch.loggers as lightning_log
import lightning.pytorch.callbacks as lightning_call

import lps_ml.datasets as ml_db
import lps_ml.model as ml_model
import lps_ml.utils.device as ml_device
import lps_ml.utils.general as ml_utils

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

    parser = argparse.ArgumentParser(description="Train an MLP classifier on MNIST.")
    parser.add_argument("--data-dir", type=str, default="/data",
                        help="Directory to store MNIST data.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--binary", action="store_true",
                        help="Train binary classifier (even=0, odd=1) instead of 10-class MNIST.")
    parser.add_argument("--model", type=str, choices=["mlp", "cnn"], default="mlp",
                        help="Model type to train: 'mlp' or 'cnn'.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    dm = ml_db.MNIST(data_dir=args.data_dir,
                     batch_size=args.batch_size,
                     binary=args.binary)
    dm.prepare_data = lambda: None

    if args.model == "mlp":
        model = ml_model.MLP(
            input_shape=dm.get_sample_shape(),
            hidden_channels=[64],
            n_targets=dm.get_n_targets(),
            loss_fn=torch.nn.BCEWithLogitsLoss if args.binary else torch.nn.CrossEntropyLoss,
            lr=args.lr,
        )
    elif args.model == "cnn":
        model = ml_model.CNN2D(
            input_shape=dm.get_sample_shape(),
            conv_n_neurons=[32, 64],
            n_targets=dm.get_n_targets(),
            loss_fn=torch.nn.BCEWithLogitsLoss if args.binary else torch.nn.CrossEntropyLoss,
            lr=args.lr,
        )
    else:
        raise ValueError("Suported modules are 'mlp' or 'cnn'")

    checkpoint_cb = lightning_call.ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename=(
            f"mnist-{args.model}-"
            f"{'binary' if args.binary else 'multi'}-"
            f"{{epoch:02d}}-{{val_loss:.3f}}"
        ),
    )
    early_stop_cb = lightning_call.EarlyStopping(monitor="val_loss", patience=4, mode="min")

    logger = lightning_log.TensorBoardLogger(
        "logs",
        name=f"mnist_{args.model}_{'binary' if args.binary else 'multi'}"
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
    test_acc  = _evaluate_accuracy(model, dm.test_dataloader())

    print(f"{'='*60}")
    print(f"Model: {args.model.upper()} | "
          f"Mode: {'Binary (Even/Odd)' if args.binary else 'Multiclass (0–9)'}")
    print(f"Train accuracy:      {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    _main()
