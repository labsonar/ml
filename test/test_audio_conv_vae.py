import os
import random
import argparse
import torch
import torchvision
import matplotlib.pyplot as plt

import lightning
import lightning.pytorch.callbacks as lightning_call

import lps_utils.quantities as lps_qty
import lps_sp.signal as lps_sig
import lps_sp.acoustical.broadband as lps_bb
import lps_sp.acoustical.analysis as lps_analysis
import lps_ml.utils.general as ml_utils
import lps_ml.datasets as ml_db
import lps_ml.model as ml_model
import lps_ml.core.cv as ml_cv
import lps_ml.audio_processors as ml_procs
import lps_ml.datasets.selection as ml_sel
import lps_ml.model.audio_vae as lps_audio_vae

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

        if "val_loss" in metrics:
            self.val_losses.append(
                metrics["val_loss"].detach().cpu().item()
            )

    def on_fit_end(self, trainer, pl_module):

        plt.figure()
        plt.semilogx(self.train_losses, label="Train Loss")
        # plt.semilogx(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
        plt.close()


class VAEComparisonCallback(lightning.Callback):

    def __init__(self, every_n_epochs: int = 5, fs: int = 16000):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.fs = fs

    @staticmethod
    def _save_audio(tensor, fs, filename):

        tensor = torch.clamp(tensor, -1.0, 1.0)
        tensor_int16 = (tensor * 32767.0).to(torch.int16)
        signal_np = tensor_int16.detach().cpu().numpy()

        lps_sig.save_wav(signal_np, fs, filename)



def _main():
    parser = argparse.ArgumentParser(
        description="Test AudioFolder dataset"
    )
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--pqmf_bands", type=int, default=8)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--ratios",
                        type=int,
                        nargs='+',
                        default=[8, 8, 4, 4],
                        help="Lista de fatores de downsampling para o ruído"
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_kl", type=float, default=0.1)
    parser.add_argument("--stft_factor", type=float, default=1)
    parser.add_argument("--mel_factor", type=float, default=1)
    parser.add_argument("--lofar_factor", type=float, default=0)
    parser.add_argument("--demon_factor", type=float, default=0)
    parser.add_argument("--output_dir", type=str, default="./result/audio_conv_vae")
    parser.add_argument("input_dir", type=str, help="Root directory containing class subfolders")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    fs=lps_qty.Frequency.khz(16)
    n_samples=int(2**17)
    overlap=0
    # n_samples=int(2**17)    #8.192s
    # overlap=int(2**16)      #4.096s

    dm = ml_db.AudioFolder(
        file_processor=ml_procs.SampleProcessor(
                fs_out=fs,
                n_samples=n_samples,
                overlap=overlap,
                pipelines=[ml_procs.ToFloatConverter()]
            ),
        cv=ml_cv.FiveByTwo(),
        input_dir=args.input_dir,
        batch_size=16,
        num_workers=1
    )

    model = lps_audio_vae.CONV_VAE(
        n_bands=args.pqmf_bands,
        capacity=args.capacity,
        latent_dim=args.latent_dim,
        beta_kl=args.beta_kl,
        stft_factor=args.stft_factor,
        mel_factor=args.mel_factor,
        lofar_factor=args.lofar_factor,
        demon_factor=args.demon_factor,
        lr=args.lr,
        ratios=args.ratios,
    )

    early_stop_callback = lightning_call.EarlyStopping(
        monitor="val/loss",
        min_delta=0.001,
        patience=300,
        verbose=True,
        mode="min"
    )

    loss_plot_callback = LossPlotCallback(output_dir=output_dir)

    checkpoint_callback = lightning_call.ModelCheckpoint(
        dirpath=output_dir,
        filename="audio_vae-{epoch:04d}-{val_loss:.4f}",
        monitor="val/loss",
        save_top_k=1,      # salva o melhor modelo
        mode="min",
        save_last=True     # salva também o último
    )

    trainer = lightning.Trainer(
        max_epochs=10000,
        accelerator="auto",
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            loss_plot_callback
        ],
        check_val_every_n_epoch=1
    )

    trainer.fit(model, datamodule=dm)
    print("Treino concluído. Gerando reconstruções finais...")

    model.eval()
    traindata = dm.train_dataloader()
    batch = next(iter(traindata))
    x, _ = batch
    x = x[:1].to(model.device)

    with torch.no_grad():
        y = model(x)

    x = x[0].detach().cpu().squeeze()
    y = y[0].detach().cpu().squeeze()

    wav_in = os.path.join(output_dir, "in.wav")
    wav_out = os.path.join(output_dir, "out.wav")

    VAEComparisonCallback._save_audio(x, fs, wav_in)
    VAEComparisonCallback._save_audio(y, fs, wav_out)

    x = x.numpy()
    y = y.numpy()

    psd_filename = os.path.join(output_dir, "psd.png")
    demon_filename = os.path.join(output_dir, "demon.png")
    lofar_filename = os.path.join(output_dir, "lofar.png")
    time_filename = os.path.join(output_dir, "time.png")

    noises=[x, y]
    labels=["Input", "Reconstructed"]

    lps_bb.plot_psds(
        filename=psd_filename,
        noises=noises,
        labels=labels,
        fs=fs,
        window_size=1024*16,
        overlap=0.5,
    )

    lps_bb.plot_demon_lines(
        filename=demon_filename,
        signals=noises,
        labels=labels,
        fs=fs,
    )

    lps_analysis.plot_spectral_analysis(
        filename=lofar_filename,
        signals=noises,
        labels=labels,
        fs=fs,
    )

    lps_analysis.plot_in_time(
        filename=time_filename,
        signals=noises,
        labels=labels,
        fs=fs,
        zoom_samples=1024*4
    )

if __name__ == "__main__":
    _main()
