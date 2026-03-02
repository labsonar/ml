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

    @classmethod
    def from_cnn(cls,
                input_shape: typing.Iterable[int],
                latent_dim: int,
                hidden_channels: typing.Iterable[int],
                kernel_size: int = 15,
                beta: float = 1.0):
        """
        Named constructor que cria uma VAE baseada em CNN 1D para áudio no domínio do tempo.
        """

        assert len(input_shape) == 1, "from_cnn suporta apenas áudio 1D"

        input_length = input_shape[0]

        # =========================
        # Encoder
        # =========================
        encoder_layers = []
        in_ch = 1
        current_length = input_length

        for out_ch in hidden_channels:
            encoder_layers.append(
                torch.nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2
                )
            )
            encoder_layers.append(torch.nn.BatchNorm1d(out_ch))
            encoder_layers.append(torch.nn.ReLU())
            in_ch = out_ch
            current_length = (current_length + 1) // 2  # aproximação para stride=2

        encoder_layers.append(torch.nn.Flatten())

        encoder_conv = torch.nn.Sequential(*encoder_layers)

        encoder_fc = torch.nn.Linear(
            hidden_channels[-1] * current_length,
            latent_dim * 2
        )

        encoder = torch.nn.Sequential(
            torch.nn.Unflatten(1, (1, input_length)),
            encoder_conv,
            encoder_fc
        )

        # =========================
        # Decoder
        # =========================
        decoder_input_dim = hidden_channels[-1] * current_length

        decoder_fc = torch.nn.Linear(latent_dim, decoder_input_dim)

        decoder_layers = []
        hidden_rev = list(reversed(hidden_channels))

        in_ch = hidden_rev[0]

        decoder_layers.append(
            torch.nn.Unflatten(1, (in_ch, current_length))
        )

        for out_ch in hidden_rev[1:]:
            decoder_layers.append(
                torch.nn.ConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    output_padding=1
                )
            )
            decoder_layers.append(torch.nn.BatchNorm1d(out_ch))
            decoder_layers.append(torch.nn.ReLU())
            in_ch = out_ch

        # camada final
        decoder_layers.append(
            torch.nn.Conv1d(
                in_ch,
                1,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            )
        )
        decoder_layers.append(torch.nn.Tanh())

        decoder = torch.nn.Sequential(
            decoder_fc,
            *decoder_layers,
            torch.nn.Flatten()
        )

        return cls(
            encoder=encoder,
            decoder=decoder,
            input_shape=input_shape,
            latent_dim=latent_dim,
            beta=beta
        )