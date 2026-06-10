"""
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scipy

import torch
import torch.utils.data as torch_data

import lightning
import lightning.pytorch.loggers as lightning_log
import lightning.pytorch.callbacks as lightning_call

import lps_utils.quantities as lps_qty
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.acoustical.analysis as lps_analysis

import lps_ml.utils.device as ml_device
import lps_ml.model as ml_model
import lps_ml.core.cv as ml_cv
import lps_ml.audio_processors as ml_procs
import lps_ml.datasets as ml_db
import lps_ml.utils.general as ml_utils
import lps_ml.datasets.synthetic as ml_synthetic

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

class SampleReconstructionCallback(lightning.Callback):
    """
    Saves reconstructed audio samples (.wav) using scipy.

    Outputs:
        - sample_i_original.wav
        - sample_i_generated.wav
    """

    def __init__(self, output_dir: str, vae_encoder, n_samples: int = 5, sample_rate: int = 16000):
        super().__init__()
        self.output_dir = output_dir
        self.vae_encoder = vae_encoder
        self.n_samples = n_samples
        self.sample_rate = sample_rate

    def _to_int16(self, x: np.ndarray) -> np.ndarray:
        """Normalize and convert to int16."""
        x = np.squeeze(x)

        max_val = np.max(np.abs(x)) + 1e-8
        x = x / max_val

        return (x * 32767).astype(np.int16)

    def on_fit_end(self, trainer, pl_module):

        device = pl_module.device
        dm = trainer.datamodule

        loader = dm.val_dataloader()
        batch = next(iter(loader))


        data, _ = batch
        x1 = data[0]
        x2 = data[1]

        x1 = x1[:self.n_samples].to(device)
        x2 = x2[:self.n_samples].to(device)

        with torch.no_grad():
            generated_latent = pl_module.sample(
                cond=x1
            )

        save_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(save_dir, exist_ok=True)

        for i in range(self.n_samples):

            z_orig = x1[i].detach().cpu().numpy()
            z_target = x2[i].detach().cpu().numpy()
            z_gen = generated_latent[i].detach().cpu().numpy()

            wav_orig = self.vae_encoder.decode(z_orig)
            wav_target = self.vae_encoder.decode(z_target)
            wav_gen = self.vae_encoder.decode(z_gen)

            signals=[wav_orig.reshape(-1), wav_target.reshape(-1), wav_gen.reshape(-1)]
            labels=["Condicionante", "Alvo", "Gerado"]

            wav_orig = self._to_int16(wav_orig)
            wav_target = self._to_int16(wav_target)
            wav_gen = self._to_int16(wav_gen)

            scipy.wavfile.write(
                os.path.join(save_dir, f"sample_{i}_original.wav"),
                self.sample_rate,
                wav_orig
            )

            scipy.wavfile.write(
                os.path.join(save_dir, f"sample_{i}_target.wav"),
                self.sample_rate,
                wav_target
            )

            scipy.wavfile.write(
                os.path.join(save_dir, f"sample_{i}_generated.wav"),
                self.sample_rate,
                wav_gen
            )

            lps_bb.plot_psds(
                filename=os.path.join(save_dir, f"sample_{i}_psds.png"),
                noises=signals,
                labels=labels,
                fs=lps_qty.Frequency.hz(self.sample_rate),
                window_size=1024*16,
                overlap=0.5,
            )

            lps_analysis.plot_spectral_analysis(
                filename=os.path.join(save_dir, f"sample_{i}_lofar.png"),
                signals=signals,
                labels=labels,
                fs=lps_qty.Frequency.hz(self.sample_rate),
                analysis=lps_analysis.SpectralAnalysis.LOFAR,
            )

            lps_analysis.plot_spectral_analysis(
                filename=os.path.join(save_dir, f"sample_{i}_mel.png"),
                signals=signals,
                labels=labels,
                fs=lps_qty.Frequency.hz(self.sample_rate),
                analysis=lps_analysis.SpectralAnalysis.MELGRAM,
            )

def _main():
    """Main function for the dataset info tables."""

    builder = ml_db.IemanjaBuilder(vae_exclusive=True)

    parser = argparse.ArgumentParser(description="Train an LDM on simple_version of iemanja.")
    parser.add_argument("--ldm-steps", type=int, default=300, help="Denoising steps for LDM.")
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
        help="Maximum number of training epochs."
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=128,
        help="Base number of channels in the U-Net."
    )

    parser.add_argument(
        "--channel-ratios",
        type=int,
        nargs="+",
        default=[1, 2, 4],
        help="Channel multipliers for each U-Net level. Example: --channel-ratios 1 2 4 8"
    )

    parser.add_argument(
        "--num-res-blocks",
        type=int,
        default=2,
        help="Number of residual blocks per U-Net level."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate."
    )

    parser.add_argument(
        "--loss",
        type=str,
        choices=[loss.name for loss in ml_model.LDMLoss],
        default=ml_model.LDMLoss.MSE.name,
        help="Loss function"
    )

    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
        help="Minimum improvement required to reset the early stopping counter."
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=300,
        help="Number of validation epochs without improvement before stopping."
    )
    parser.add_argument("--output-dir", type=str, default="./result/ldm")

    builder.add_argparse_args(parser=parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    dm = builder.paired_from_argparse_args(args)
    vae_encoder = dm.file_processor.pipelines[-1]
    dm.setup()

    print(ml_utils.format_header(60,"Dataset description"))
    print(dm.to_compile_df())
    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"Training"))
    print()
    print(dm.to_df())

    train_loader = dm.train_dataloader()

    x, y = next(iter(train_loader))

    if isinstance(x, list):
        print("x: ", len(x))
        for i in x:
            print("\ti: ", i.shape)

        batch_size = x[0].shape[0]
        latent_channels = x[0].shape[1]
        latent_length = x[0].shape[2]

    else:
        print("x shape:", x.shape)
        print("y shape:", y.shape)

        batch_size = x.shape[0]
        latent_channels = x.shape[1]
        latent_length = x.shape[2]

    device = ml_device.get_available_device()

    # -------------------------
    # Infer latent dimensions
    # -------------------------

    print("batch_size:", batch_size)
    print("latent_channels:", latent_channels)
    print("latent_length:", latent_length)

    model = ml_model.LatentDiffusionModel(
        in_channels=latent_channels,
        base_channels=args.base_channels,
        channel_ratios=args.channel_ratios,
        num_res_blocks=args.num_res_blocks,
        timesteps=args.ldm_steps,
        lr=args.lr,
        loss = ml_model.LDMLoss[args.loss.upper()]
    )

    callbacks = [
        lightning_call.ModelCheckpoint(
            dirpath=args.output_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best"
        ),
        lightning_call.EarlyStopping(
            monitor="val/loss",
            min_delta=args.early_stopping_min_delta,
            patience=args.early_stopping_patience,
            verbose=True,
            mode="min"
        ),
        lightning_call.LearningRateMonitor(logging_interval='epoch'),
        LossPlotCallback(args.output_dir),
        SampleReconstructionCallback(args.output_dir, vae_encoder)
    ]

    trainer = lightning.Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        accelerator="auto",
        check_val_every_n_epoch=1
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    _main()
