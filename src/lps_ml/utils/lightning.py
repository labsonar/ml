import os
import typing
import numpy as np
import matplotlib.pyplot as plt

import torch
import lightning
import lightning.pytorch.callbacks as lightning_call

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_sig
import lps_ml.utils.sonar_loss as ml_loss


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

class SaveAudioSamples(lightning.Callback):

    def __init__(
        self,
        output_dir: str,
        n_samples: int = 5,
        sample_rate: lps_qty.Frequency = lps_qty.Frequency.khz(16),
    ):
        super().__init__()
        self.output_dir = os.path.join(output_dir, "audio_samples")
        self.n_samples = n_samples
        self.sample_rate = sample_rate

    def on_fit_end(self, trainer, pl_module):
        os.makedirs(self.output_dir, exist_ok=True)

        pl_module.eval()

        dataloader = trainer.val_dataloaders
        if isinstance(dataloader, list):
            dataloader = dataloader[0]

        saved = 0

        device = pl_module.device

        stft_loss = ml_loss.MultiResolutionLoss[ml_loss.STFT]([
            ml_loss.STFTConfig(4096, 2048),
        ])

        mel_loss = ml_loss.MultiResolutionLoss[ml_loss.Mel]([
            ml_loss.MelConfig(4096, 2048, n_mels=512),
        ])

        lofar_loss = ml_loss.MultiResolutionLoss[ml_loss.Lofar]([
            ml_loss.LofarConfig(4096, 2048),
        ])

        demon_loss = ml_loss.MultiResolutionLoss[ml_loss.Demon]([
            ml_loss.DemonConfig(1024, 512, decimate=[16, 8]),
        ])

        loss = ml_loss.SonarLoss(
            stft_factor=1.0,
            mel_factor=1.0,
            lofar_factor=1.0,
            demon_factor=1.0,
            stft_loss=stft_loss,
            mel_loss=mel_loss,
            lofar_loss=lofar_loss,
            demon_loss=demon_loss
        )

        with torch.no_grad():

            for batch in dataloader:

                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(device)
                out = pl_module(x)

                for i in range(x.shape[0]):
                    if saved >= self.n_samples:
                        return

                    output_dir = os.path.join(self.output_dir, str(saved))
                    os.makedirs(output_dir, exist_ok=True)

                    original = x[i].detach().cpu()
                    recon = out[i].detach().cpu()

                    loss.plot([original, recon], output_dir=output_dir)

                    original = original.numpy()
                    recon = recon.numpy()

                    if original.ndim == 1:
                        original = np.expand_dims(original, axis=0)

                    if recon.ndim == 1:
                        recon = np.expand_dims(recon, axis=0)

                    lps_sig.save_convert_wav(
                        filename=os.path.join(output_dir, f"sample_{saved}_original.wav"),
                        signal=original,
                        fs=self.sample_rate,
                    )

                    lps_sig.save_convert_wav(
                        filename=os.path.join(output_dir, f"sample_{saved}_recon.wav"),
                        signal=recon,
                        fs=self.sample_rate,
                    )

                    saved += 1

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
                      min_delta: float = 0.001,
                      n_audio_samples: int = 5) -> typing.List[lightning_call.Callback]:
    """ Default training callbacks. """

    checkpoint_cb = default_checkpoint(output_dir)

    return [
        # default_early_stop(patience=patience, min_delta=min_delta),
        checkpoint_cb,
        PlotMetrics(output_dir=output_dir),
        SaveAudioSamples(
            output_dir=output_dir,
            n_samples=n_audio_samples,
        ),
    ]

def default_trainer(output_dir: str,
                    max_epochs : int = 10000,
                    check_val_every_n_epoch=1,
                    patience : int = 100,
                    min_delta: float = 0.001,
                    n_audio_samples: int = 5) -> lightning.Trainer:
    """ Default trainer with common callbacks. """
    return lightning.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=default_callbacks(output_dir, patience, min_delta, n_audio_samples),
        check_val_every_n_epoch=check_val_every_n_epoch
    )
