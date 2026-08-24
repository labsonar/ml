"""
Olocum
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

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            if out.ndim == 1:
                preds = (out >= 0.5).long()
            else:
                preds = torch.argmax(out, dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    balanced_acc = sk_metrics.balanced_accuracy_score(y_true, y_pred)
    macro_f1 = sk_metrics.f1_score(y_true, y_pred, average="macro")

    return balanced_acc, macro_f1

def _main():
    """Main function for the dataset info tables."""

    builder = ml_db.IemanjaBuilder()

    parser = argparse.ArgumentParser(description="Train an MLP classifier on iara.")
    parser.add_argument("--max-epochs", type=int, default=2000,
                        help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--only-info", action="store_true", help="Only print database info")
    parser.add_argument("--paired", action="store_true", help="Only print database info")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001,
        help="Minimum improvement required to reset the early stopping counter.")
    parser.add_argument("--early-stopping-patience", type=int, default=200,
        help="Number of validation epochs without improvement before stopping.")
    parser.add_argument("--model", choices=["mlp", "cnn1d", "cnn2d"], default="cnn2d",
        help="CNN architecture to use.")
    parser.add_argument("--output_dir", type=str, default="./result/classifier",
        help="Output directory for results.")
    builder.add_argparse_args(parser=parser)
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    if args.paired:
        dm = builder.paired_from_argparse_args(args)
    else:
        dm = builder.from_argparse_args(args)

    print(ml_utils.format_header(60,"Dataset description"))
    print(dm.to_compile_df())
    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"Training"))
    print()
    print(dm.to_df())

    if args.only_info:
        dm.setup()
        dl = dm.train_dataloader()
        for x, y in dl:
            if isinstance(x, list):
                print("x: ", len(x))
                for i in x:
                    print("\ti: ", i.shape)
            else:
                print("x: ", x.shape)

            print("y: ", y.shape)
            print(y)
            break

    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_last = os.path.join(output_dir, "last.ckpt")
        model_best = os.path.join(output_dir, "best.ckpt")
        log_dir = os.path.join(output_dir, "log")

        if os.path.exists(model_last):
            dm.setup()

        else:

            if args.model == "mlp":

                model = ml_model.MLP(
                    input_shape=dm.get_sample_shape(),

                    hidden_channels=[128, 32],

                    n_targets=dm.get_n_targets(),

                    norm_layer=torch.nn.BatchNorm1d,
                    activation_layer=torch.nn.ReLU,
                    activation_output_layer=torch.nn.Sigmoid,

                    dropout=0.4,

                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )

            elif args.model == "cnn1d":
                model = ml_model.CNN1D(
                    input_shape=dm.get_sample_shape(),

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
                    n_targets=dm.get_n_targets(),
                    classification_dropout=0.4,
                    classification_norm=None,
                    classification_output_activation=torch.nn.Sigmoid,

                    # Optimization
                    lr=args.lr,
                )

            else:

                model = ml_model.CNN2D(
                    input_shape=dm.get_sample_shape(),

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
                    n_targets=dm.get_n_targets(),
                    classification_dropout=0.5,
                    classification_norm=None,
                    classification_hidden_activation=torch.nn.ReLU,
                    classification_output_activation=torch.nn.Sigmoid,

                    # Optimization
                    lr=args.lr,
                    weight_decay=args.weight_decay
                )

            checkpoint_cb = lightning_call.ModelCheckpoint(
                dirpath=log_dir,
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
                filename="best"
            )

            callbacks = [
                checkpoint_cb,
                # lightning_call.EarlyStopping(
                #     monitor="val/loss",
                #     min_delta=args.early_stopping_min_delta,
                #     patience=args.early_stopping_patience,
                #     verbose=True,
                #     mode="min"
                # ),
                lightning_call.LearningRateMonitor(logging_interval='epoch'),
            ]

            logger = lightning_log.TensorBoardLogger(log_dir, name="iara")

            trainer = lightning.Trainer(
                max_epochs=args.max_epochs,
                accelerator="auto",
                devices="auto",
                logger=logger,
                callbacks=callbacks,
            )

            trainer.fit(model, dm)
            trainer.test(model, datamodule=dm)

            shutil.copy2(checkpoint_cb.last_model_path, model_last)
            shutil.copy2(checkpoint_cb.best_model_path, model_best)


        if args.model == "mlp":
            best_model = ml_model.MLP.load_from_checkpoint(model_best)
        elif args.model == "cnn1d":
            best_model = ml_model.CNN1D.load_from_checkpoint(model_best)
        else:
            best_model = ml_model.CNN2D.load_from_checkpoint(model_best)

        best_model.eval()

        train_bal_acc, train_f1 = _evaluate_accuracy(best_model, dm.train_dataloader())
        val_bal_acc, val_f1   = _evaluate_accuracy(best_model, dm.val_dataloader())
        test_bal_acc, test_f1  = _evaluate_accuracy(best_model, dm.test_dataloader())

        metrics_df = pd.DataFrame(
            {
                "balanced_accuracy": [
                    train_bal_acc,
                    val_bal_acc,
                    test_bal_acc,
                ],
                "macro_f1": [
                    train_f1,
                    val_f1,
                    test_f1,
                ],
            },
            index=["train", "val", "test"]
        )

        metrics_df.index.name = "dataset"

        metrics_path = os.path.join(output_dir, "metrics.csv")
        metrics_df.to_csv(metrics_path)

        print(ml_utils.format_header(60))
        print()
        print(ml_utils.format_header(60,"Results"))
        print(metrics_df)
        print(ml_utils.format_header(60))

if __name__ == "__main__":
    _main()
