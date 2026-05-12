"""
Olocum
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

        x1, x2 = batch
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

    parser = argparse.ArgumentParser(description="Train an LDM on simple_version of iemanja.")
    parser.add_argument("--model", type=str, default="/data/models/v0_6M.ts")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--latent_compactness", type=int, default=1024, help="Compression of VAE.")
    parser.add_argument("--ldm-steps", type=int, default=300, help="Denoising steps for LDM.")
    parser.add_argument("--max-epochs", type=int, default=1000,
                        help="Maximum number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--output-dir", type=str, default="./result/ldm")
    parser.add_argument(
        "--dynamic_selection",
        type=str,
        default=ml_db.DynamicSelection.FIXED_ONLY.name,
        choices=[e.name for e in ml_db.DynamicSelection],
        help=(
            "Dynamic selection mode. "
            f"Options: {[e.name for e in ml_db.DynamicSelection]}"
        )
    )
    parser.add_argument(
        "--channel_selection",
        type=str,
        default=ml_db.ChannelSelection.REFERENCE_ONLY.name,
        choices=[e.name for e in ml_db.ChannelSelection],
        help=(
            "Channel selection mode. "
            f"Options: {[e.name for e in ml_db.ChannelSelection]}"
        )
    )
    args = parser.parse_args()

    dynamic_selection = ml_db.DynamicSelection[args.dynamic_selection]
    channel_selection = ml_db.ChannelSelection[args.channel_selection]

    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    n_samples=int(2**17)    #8.192s
    overlap=int(2**16)      #4.096s

    vae_encoder = ml_procs.VAEEncoder(args.model)

    dm = ml_db.IemanjaPaired(
            file_processor=ml_procs.SampleProcessor(
                    n_samples=int(n_samples/args.latent_compactness),
                    overlap=int(overlap/args.latent_compactness),
                    pipelines=[
                        ml_procs.ToFloatConverter(),
                        vae_encoder
                    ]
                ),
            cv = ml_cv.SimpleSplitCV(),
            dynamic_selection=dynamic_selection,
            channel_selection=channel_selection,
            batch_size=args.batch_size
            )
    dm.setup()

    print(ml_utils.format_header(60,"Dataset description"))
    print(dm.to_compile_df())
    print(ml_utils.format_header(60))
    print()
    print(ml_utils.format_header(60,"Training"))
    print()
    print(dm.to_df())

    dm.to_df().to_csv("./result/paired.csv")

    train_loader = dm.train_dataloader()

    x1, x2 = next(iter(train_loader))

    print("x1 shape:", x1.shape)
    print("x2 shape:", x2.shape)

    device = ml_device.get_available_device()

    # -------------------------
    # Infer latent dimensions
    # -------------------------
    batch_size = x1.shape[0]
    latent_channels = x1.shape[1]
    latent_length = x1.shape[2]

    print("batch_size:", batch_size)
    print("latent_channels:", latent_channels)
    print("latent_length:", latent_length)

    model = ml_model.LatentDiffusionModel(
        in_channels=latent_channels,
        base_channels=128,
        channel_ratios=[1, 2, 4],
        num_res_blocks=2,
        timesteps=args.ldm_steps,
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
            patience=100,
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
