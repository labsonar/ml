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
    parser.add_argument("--pqmf_bands", type=int, default=8)
    parser.add_argument("--n_harmonics", type=int, default=8)
    parser.add_argument("--nb_ratios",
                        type=int,
                        nargs='+',
                        default=[8, 8, 4, 4],
                        help="Lista de fatores de downsampling para o ruído de banda estreita"
    )
    parser.add_argument("--bb_ratios",
                        type=int,
                        nargs='+',
                        default=[8, 8, 8, 8, 4],
                        help="Lista de fatores de downsampling para o ruído de banda larga"
    )
    parser.add_argument("--stft_factor", type=float, default=1)
    parser.add_argument("--mel_factor", type=float, default=1)
    parser.add_argument("--lofar_factor", type=float, default=0)
    parser.add_argument("--demon_factor", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--resume_ckpt",
                        type=str,
                        default=None,
                        help="Checkpoint para retomar treinamento"
    )
    parser.add_argument("--pretrained_ckpt",
                        type=str,
                        default=None,
                        help="Checkpoint pré-treinado para fine-tuning"
    )

    parser.add_argument("--output_dir", type=str, default="./result/ddsp/bb")
    parser.add_argument("input_dir", type=str, help="Root directory containing class subfolders")
    args = parser.parse_args()

    if args.resume_ckpt and args.pretrained_ckpt:
        raise ValueError("Use apenas um: --resume_ckpt OU --pretrained_ckpt")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    torch.set_float32_matmul_precision('medium')
    ml_utils.set_seed()

    fs=lps_qty.Frequency.khz(16)
    n_samples=int(2**17)
    overlap=0

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

    if args.pretrained_ckpt is not None:
        model = lps_audio_vae.DDSP.load_from_checkpoint(
            args.pretrained_ckpt,
            stft_factor=args.stft_factor,
            mel_factor=args.mel_factor,
            lofar_factor=args.lofar_factor,
            demon_factor=args.demon_factor,
            lr=args.lr,
            strict=False
        )
    else:
        model = lps_audio_vae.DDSP(
            n_bands=args.pqmf_bands,
            n_harmonics=args.n_harmonics,
            nb_ratios=args.nb_ratios,
            bb_ratios=args.bb_ratios,

            stft_factor=args.stft_factor,
            mel_factor=args.mel_factor,
            lofar_factor=args.lofar_factor,
            demon_factor=args.demon_factor,
            lr=args.lr,
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

    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt)
    print("Treino concluído. Gerando reconstruções finais...")

    model.eval()
    traindata = dm.train_dataloader()
    batch = next(iter(traindata))
    x, _ = batch
    x = x[:1].to(model.device)

    with torch.no_grad():
        y, h_t = model.internal_forward(x)

    model.loss.plot([x, y], output_dir=output_dir)

    x = x[0].detach().cpu().squeeze()
    y = y[0].detach().cpu().squeeze()

    wav_in = os.path.join(output_dir, "in.wav")
    wav_out = os.path.join(output_dir, "out.wav")

    VAEComparisonCallback._save_audio(x, fs, wav_in)
    VAEComparisonCallback._save_audio(y, fs, wav_out)


    lps_analysis.plot_in_time(
        filename=os.path.join(output_dir, "h_t.png"),
        signals=[h_t],
        labels=["h(t)"],
        fs=fs,
    )

if __name__ == "__main__":
    _main()
