import os
import typing
import matplotlib.pyplot as plt

import torch
import lightning
import lightning.pytorch.callbacks as lightning_call


class PlotMetrics(lightning.Callback):
    """
    Generic callback to plot training and validation metrics.

    Args:
        metrics: list of base metric names (e.g., ["loss", "acc"])
        train_prefix: prefix for training metrics (default: "train/")
        val_prefix: prefix for validation metrics (default: "val/")
        output_dir: output directory
        log_scale: if True, uses a logarithmic scale on the Y-axis
    """

    def __init__(
        self,
        output_dir: str,
        metrics: typing.List[str] = ["loss"],
        train_prefix: str = "train/",
        val_prefix: str = "val/",
        log_scale: bool = True,
    ):
        super().__init__()

        self.metrics = metrics
        self.train_prefix = train_prefix
        self.val_prefix = val_prefix
        self.output_dir = output_dir
        self.log_scale = log_scale

        self.storage = {
            m: {"train": [], "val": []} for m in metrics
        }

    def _extract_metric(self, metrics_dict, key):
        if key in metrics_dict:
            return metrics_dict[key].detach().cpu().item()
        return None

    def on_train_epoch_end(self, trainer, pl_module):
        metrics_dict = trainer.callback_metrics

        for m in self.metrics:
            key = f"{self.train_prefix}{m}"
            value = self._extract_metric(metrics_dict, key)
            if value is not None:
                self.storage[m]["train"].append(value)

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics_dict = trainer.callback_metrics

        for m in self.metrics:
            key = f"{self.val_prefix}{m}"
            value = self._extract_metric(metrics_dict, key)
            if value is not None:
                self.storage[m]["val"].append(value)

    def on_fit_end(self, trainer, pl_module):

        for m in self.metrics:
            plt.figure()

            train_values = self.storage[m]["train"]
            val_values = self.storage[m]["val"]

            if train_values:
                if self.log_scale:
                    plt.semilogx(train_values, label=f"Train {m}")
                else:
                    plt.plot(train_values, label=f"Train {m}")

            if val_values:
                if self.log_scale:
                    plt.semilogx(val_values, label=f"Val {m}")
                else:
                    plt.plot(val_values, label=f"Val {m}")

            plt.xlabel("Epoch")
            plt.ylabel(m)
            plt.legend()
            plt.grid(True)

            plt.savefig(os.path.join(self.output_dir, f"{m}_curve.png"))
            plt.close()

def default_early_stop(patience : int = 100, min_delta: float = 0.001) -> lightning_call.Callback:
    """ Early stopping callback to monitor the "value/loss" metric. """
    return lightning_call.EarlyStopping(
        monitor="val/loss",
        min_delta=min_delta,
        patience=patience,
        verbose=True,
        mode="min"
    )

def default_checkpoint(output_dir: str) -> lightning_call.ModelCheckpoint:
    """ Model checkpoint callback to save the best model based on "val/loss" and also the last. """
    return lightning_call.ModelCheckpoint(
        dirpath=output_dir,
        filename="best",
        monitor="val/loss",
        save_top_k=1,
        mode="min",
        save_last=True
    )

def default_callbacks(output_dir: str,
                      patience : int = 100,
                      min_delta: float = 0.001) -> typing.List[lightning_call.Callback]:
    """ Default training callbacks. """

    checkpoint_cb = default_checkpoint(output_dir)

    return [
        # default_early_stop(patience=patience, min_delta=min_delta),
        checkpoint_cb,
        PlotMetrics(output_dir=output_dir),
    ]

def default_trainer(output_dir: str,
                    max_epochs : int = 10000,
                    check_val_every_n_epoch=1,
                    patience : int = 100,
                    min_delta: float = 0.001) -> lightning.Trainer:
    """ Default trainer with common callbacks. """
    return lightning.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=default_callbacks(output_dir, patience, min_delta),
        check_val_every_n_epoch=check_val_every_n_epoch
    )
