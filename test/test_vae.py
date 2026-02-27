import os
import torch
import torchvision

import lightning
import lightning.pytorch.callbacks as lightning_call

import lps_ml.datasets as ml_db
import lps_ml.model as ml_model

OUTPUT_DIR = "./result/vae"

class VAEComparisonCallback(lightning.Callback):

    def __init__(self, every_n_epochs: int = 5):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):

        if (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        pl_module.eval()

        n_samples = 16

        batch = next(iter(trainer.datamodule.val_dataloader()))
        x, _ = batch
        x = x[:n_samples].to(pl_module.device)

        with torch.no_grad():
            recon, _, _ = pl_module(x)

            samples = pl_module.sample(n_samples, pl_module.device)

        combined = torch.cat([x, recon, samples], dim=0)
        grid = torchvision.utils.make_grid(combined, nrow=n_samples)
        torchvision.utils.save_image(
            grid,
            os.path.join(OUTPUT_DIR, f"step_epoch_{trainer.current_epoch}.png")
        )
        pl_module.train()

def _main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dm = ml_db.MNIST(
        data_dir="/data",
        batch_size=128
    )

    model = ml_model.VAE.from_mlp(
        input_shape=[1, 28, 28],
        hidden_dims=[512],
        latent_dim=256,
        beta=2.0
    )

    early_stop_callback = lightning_call.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=10,
        verbose=True,
        mode="min"
    )

    trainer = lightning.Trainer(
        max_epochs=200,
        accelerator="auto",
        callbacks=[
            VAEComparisonCallback(5),
            early_stop_callback
        ],
        check_val_every_n_epoch=1
    )

    trainer.fit(model, datamodule=dm)

    print("Treino concluído. Gerando amostras finais...")
    model.eval()
    final_grid = model.sample(64, model.device)
    torchvision.utils.save_image(
        torchvision.utils.make_grid(final_grid, nrow=8),
        os.path.join(OUTPUT_DIR, "mnist_generated.png")
    )

if __name__ == "__main__":
    _main()