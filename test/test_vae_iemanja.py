import os
import random
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

OUTPUT_DIR = "./result/vae_iemanja"


class LossPlotCallback(lightning.Callback):

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        if "train_loss" in metrics:
            self.train_losses.append(
                metrics["train_loss"].detach().cpu().item()
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
        plt.semilogx(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
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

    def process_batch(self, model, input_data, tag):

        with ml_utils.evaluating(model):
            with torch.no_grad():
                recon, _, _ = model(input_data)

        for i in range(input_data.shape[0]):

            x_in = input_data[i].detach().cpu().squeeze()
            x_out = recon[i].detach().cpu().squeeze()

            wav_filename = os.path.join(
                OUTPUT_DIR,
                f"{tag}_sample_{i}.wav"
            )
            wav_in = os.path.join(
                OUTPUT_DIR,
                f"{tag}_sample_{i}_in.wav"
            )
            psd_filename = os.path.join(
                OUTPUT_DIR,
                f"{tag}_sample_{i}_psd.png"
            )
            demon_filename = os.path.join(
                OUTPUT_DIR,
                f"{tag}_sample_{i}_demon.png"
            )
            lofar_filename = os.path.join(
                OUTPUT_DIR,
                f"{tag}_sample_{i}_lofar.png"
            )

            self._save_audio(x_in, self.fs, wav_in)
            self._save_audio(x_out, self.fs, wav_filename)

            x_in = x_in.numpy()
            x_out = x_out.numpy()

            lps_bb.plot_psds(
                filename=psd_filename,
                noises=[x_in, x_out],
                labels=["Input", "Reconstructed"],
                fs=lps_qty.Frequency.hz(self.fs),
                window_size=1024*16,
                overlap=0.5,
            )

            lps_bb.plot_demon_lines(
                filename=demon_filename,
                signals=[x_in, x_out],
                labels=["Input", "Reconstructed"],
                fs=lps_qty.Frequency.hz(self.fs),
            )

            lps_analysis.plot_spectral_analysis(
                filename=lofar_filename,
                signals=[x_in, x_out],
                labels=["Input", "Reconstructed"],
                fs=lps_qty.Frequency.hz(self.fs),
            )

    def generate_reconstructions(
        self,
        model,
        dataloader,
        n_samples=2,
        tag="reconstructions"
    ):

        batch = next(iter(dataloader))
        x, _ = batch

        k = min(n_samples, x.shape[0])
        indices = random.sample(range(x.shape[0]), k=k)

        x_selected = x[indices].to(model.device)

        self.process_batch(model, x_selected, tag=tag)

    def generate_from_latent_noise(
        self,
        model,
        n_samples=5,
        epoch_tag="latent"
    ):

        with ml_utils.evaluating(model):
            with torch.no_grad():
                generated = model.sample(
                    num_samples=n_samples,
                    device=model.device
                )

        for i in range(n_samples):

            x_gen = generated[i].detach().cpu().squeeze()

            filename = os.path.join(
                OUTPUT_DIR,
                f"{epoch_tag}_generated_{i}.wav"
            )

            self._save_audio(x_gen, self.fs, filename)

    def on_train_epoch_end(self, trainer, pl_module):

        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        val_loader = trainer.datamodule.val_dataloader()

        self.generate_reconstructions(
            model=pl_module,
            dataloader=val_loader,
            n_samples=2,
            tag=f"epoch_{trainer.current_epoch}"
        )


def _main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    fs=lps_qty.Frequency.khz(16)
    n_samples=int(2**16)
    overlap=int(2**15)

    dm = ml_db.AudioFolder(
        file_processor=ml_procs.SampleProcessor(
                fs_out=fs,
                n_samples=n_samples,
                overlap=overlap,
                pipelines=[ml_procs.ToFloatConverter()]
            ),
        input_dir="/data/datatest",
        selection = ml_sel.Selector(ml_sel.LabelTarget(column="Class", values= ["cargo"])),
        batch_size=8,
        num_workers=1
    )

    # dm = ml_db.Iemanja(
    #         file_processor=ml_procs.TimeProcessor(
    #                 fs_out=fs,
    #                 duration=duration,
    #                 overlap=overlap,
    #                 pipelines=[ml_procs.ToFloatConverter()]
    #             ),
    #         cv = ml_cv.FiveByTwo(),
    #         dynamic_selection=ml_db.DynamicSelection.FIXED_ONLY,
    #         channel_selection=ml_db.ChannelSelection.REFERENCE_ONLY,
    #         batch_size=16,
    #         num_workers=1
    #         )

    model = ml_model.VAE.from_mlp(
        input_shape=[n_samples],
        hidden_dims=[512],
        latent_dim=128,
        beta=1.0
    )

    early_stop_callback = lightning_call.EarlyStopping(
        monitor="val_loss",
        min_delta=0.1,
        patience=10,
        verbose=True,
        mode="min"
    )

    loss_plot_callback = LossPlotCallback()

    vae_comp = VAEComparisonCallback(5)

    trainer = lightning.Trainer(
        max_epochs=200,
        accelerator="auto",
        callbacks=[
            vae_comp,
            early_stop_callback,
            loss_plot_callback
        ],
        check_val_every_n_epoch=1
    )

    print("Generating reconstructions BEFORE training (random weights)...")
    dm.setup()
    val_loader = dm.val_dataloader()

    vae_comp.generate_reconstructions(
        model=model,
        dataloader=val_loader,
        n_samples=2,
        tag="before_training"
    )

    vae_comp.generate_from_latent_noise(
        model=model,
        n_samples=5,
        epoch_tag="before_training_latent"
    )

    trainer.fit(model, datamodule=dm)
    print("Treino concluído. Gerando reconstruções finais...")

    vae_comp.generate_reconstructions(
        model=model,
        dataloader=val_loader,
        n_samples=5,
        tag="after_training"
    )

    vae_comp.generate_from_latent_noise(
        model=model,
        n_samples=10,
        epoch_tag="after_training_latent"
    )


if __name__ == "__main__":
    _main()
