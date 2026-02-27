import typing
import functools
import torch
import torch.nn
import lightning

import lps_ml.model.mlp as ml_mlp

class VAE(lightning.LightningModule):

    def __init__(self,
                 encoder: lightning.LightningModule,
                 decoder: lightning.LightningModule,
                 input_shape: typing.Iterable[int],
                 latent_dim: int,
                 beta: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        combined = self.encoder(x)
        mu, logvar = torch.chunk(combined, 2, dim=-1)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z).view(-1, *self.input_shape)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)

    def _shared_step(self, batch):
        x, _ = batch
        recon_x, mu, logvar = self.forward(x)

        recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
        # recon_loss = torch.nn.functional.binary_cross_entropy(
        #     recon_x.view(x.size(0), -1),
        #     x.view(x.size(0), -1),
        #     reduction="sum"
        # )
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + self.beta * kld_loss

        return {
            "loss": loss,
            "recon": recon_loss / x.size(0),
            "kl": kld_loss / x.size(0)
        }

    def training_step(self, batch, batch_idx):
        metrics = self._shared_step(batch)

        self.log("train_loss", metrics["loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_recon", metrics["recon"], on_epoch=True)
        self.log("train_kl", metrics["kl"], on_epoch=True)

        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self._shared_step(batch)

        self.log("val_loss", metrics["loss"], prog_bar=True, sync_dist=True)
        self.log("val_recon", metrics["recon"], sync_dist=True)
        self.log("val_kl", metrics["kl"], sync_dist=True)

        return metrics["loss"]


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


    @classmethod
    def from_mlp(cls,
                 input_shape: typing.Iterable[int],
                 hidden_dims: typing.Iterable[int],
                 latent_dim: int,
                 beta: float = 1.0):
        """
        Named constructor que cria uma VAE baseada em MLPs.
        """
        encoder = ml_mlp.MLP(
            input_shape=input_shape,
            hidden_channels=hidden_dims,
            n_targets=latent_dim*2,
            norm_layer = None,
            activation_layer = torch.nn.ReLU,
            activation_output_layer = None,
        )

        input_dim = functools.reduce(lambda x, y: x * y, input_shape)

        decoder = ml_mlp.MLP(
            input_shape=latent_dim,
            hidden_channels=list(reversed(hidden_dims)),
            n_targets=input_dim,
            norm_layer = None,
            activation_layer = torch.nn.ReLU,
            activation_output_layer =torch.nn.Tanh
        )

        return cls(encoder = encoder,
                   decoder = decoder,
                   input_shape = input_shape,
                   latent_dim = latent_dim,
                   beta = beta)
