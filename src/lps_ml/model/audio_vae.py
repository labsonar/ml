import typing
import torch
import lightning

import lps_utils.quantities as lps_qty
import lps_ml.model.blocks.stack1d as ml_stack1d
import lps_ml.utils.pqmf as ml_pqmf
import lps_ml.model.blocks.ddsp as ml_ddsp

class ConvEncoder(ml_stack1d.ConvEncoder):
    """
    Variational convolutional encoder.

    The encoder consists of:
        Conv1DFeatureExtractor -> Conv1d projection

    The projection outputs 2 * latent_dim channels, corresponding to
    the mean and log-variance of the latent distribution.
    """

    def __init__(
        self,
        in_channels: int,
        channels: typing.List[int],
        latent_dim: int,

        kernel_size: typing.Union[int, typing.List[int]] = 7,
        stride: typing.Union[int, typing.List[int]] = 2,
        dilation: typing.Union[int, typing.List[int]] = 1,

        activation: typing.Callable = torch.nn.LeakyReLU,
        norm: typing.Optional[typing.Callable] = torch.nn.BatchNorm1d,
        dropout: float = 0.0,
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            project_dim=2 * latent_dim,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )

        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor):
        latent_params = super().forward(x)
        mean, logvar = torch.chunk(latent_params, 2, dim=1)
        return mean, logvar

class MultiSTFTLoss(torch.nn.Module):

    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[256, 512, 128]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes

    def stft(self, x, fft_size, hop):
        return torch.stft(
            x.squeeze(1),
            n_fft=fft_size,
            hop_length=hop,
            win_length=fft_size,
            return_complex=True
        )

    def forward(self, x, y):
        loss = 0.0

        for fft, hop in zip(self.fft_sizes, self.hop_sizes):
            X = self.stft(x, fft, hop)
            Y = self.stft(y, fft, hop)

            mag_X = torch.abs(X)
            mag_Y = torch.abs(Y)

            # spectral convergence
            sc = torch.norm(mag_X - mag_Y) / (torch.norm(mag_X) + 1e-7)
            loss += sc

            # # log magnitude
            # log_mag = torch.mean(torch.abs(torch.log(mag_X + 1e-7) - torch.log(mag_Y + 1e-7)))
            # loss += sc + log_mag

        return loss

class DDSP_VAE(lightning.LightningModule):

    def __init__(
        self,

        n_bands: int = 8,
        capacity=32,
        latent_dim=64,

        kernel=3,
        ratios=[4, 4, 4, 2],

        noise_ratios=[8, 8, 4, 4],
        noise_bands=5,
        n_noise_channels=1,

        n_harmonics=12,
        sample_rate: lps_qty.Frequency = lps_qty.Frequency.khz(16),

        beta_kl=1e-4,
        lr=1e-4,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.pqmf = ml_pqmf.PQMF(n_bands)

        in_ratios = [1] + ratios
        in_layers = len(in_ratios)
        enc_channels = [capacity * (2**i) for i in range(in_layers)]
        dec_channels = list(reversed(enc_channels))

        self.encoder = ConvEncoder(
            in_channels=n_bands,
            channels=enc_channels,
            latent_dim=latent_dim,
            kernel_size=kernel,
            stride=in_ratios,
            dilation=[1 + (2*i) for i in range(in_layers)],
        )

        self.decoder = ml_stack1d.ConvFeatureReconstructor(
            in_channels=latent_dim,
            adapt_channels=dec_channels[0],
            channels=dec_channels[1:],
            up_factors=ratios,
        )

        self.bb = ml_ddsp.Broadband(
            in_channels=dec_channels[-1],
            hidden_size=dec_channels[-1],
            out_channels=n_bands,
            ratios=noise_ratios,
            noise_bands=noise_bands,
            n_noise_channels=n_noise_channels,
        )

        noise_ratios=[8, 8, 4, 4, 4]
        harmonic_head = ml_ddsp.BroadbandHarmonicModulatorHead(
            in_channels=dec_channels[-1],
            channels=[dec_channels[-1] for i in range(len(noise_ratios))],
            n_harmonics=n_harmonics,
            stride=noise_ratios,
        )

        self.bb_mod = ml_ddsp.BroadbandHarmonicModulator(
            head=harmonic_head,
            sample_rate=sample_rate,
            samples_per_frame=n_bands,
            f0_min=lps_qty.Frequency.rpm(40),
            f0_max=lps_qty.Frequency.rpm(200),
        )

        noise_ratios=[4, 4, 4]
        nb_head = ml_ddsp.NarrowbandHead(
            in_channels=dec_channels[-1],
            channels=[dec_channels[-1] for i in range(len(noise_ratios))],
            n_freqs=16,
            stride=noise_ratios,
        )

        self.nb = ml_ddsp.Narrowband(
            head=nb_head,
            sample_rate=sample_rate,
            samples_per_frame=n_bands,
        )

        self.stft_loss = MultiSTFTLoss()

    @staticmethod
    def _reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    @staticmethod
    def _kl_loss(mean, logvar):
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    def forward(self, x):
        y, _, _, _, _, _ = self.detailed_forward(x)
        return y

    def detailed_forward(self, x):
        """
        x: (B, 1, T)
        """
        print("input (B, 1, T): ", x.shape)

        x_sub = self.pqmf(x)
        print("x_sub: ", x_sub.shape)

        mean, logvar = self.encoder(x_sub)
        print("mean: ", mean.shape)
        print("logvar: ", logvar.shape)

        z = DDSP_VAE._reparameterize(mean, logvar)
        print("z: ", z.shape)

        features = self.decoder(z)
        print("features: ", features.shape)

        ship_bb_noise = self.bb(features)
        print("ship_bb_noise: ", ship_bb_noise.shape)
        ship_bb_noise = self.pqmf.reverse(ship_bb_noise)
        print("ship_bb_noise pqmf: ", ship_bb_noise.shape)

        ship_bb_modulation = self.bb_mod(features)
        print("ship_bb_modulation: ", ship_bb_modulation.shape)

        ship_nb_noise = self.nb(features)
        print("ship_nb_noise: ", ship_nb_noise.shape)

        y = ship_bb_noise * ship_bb_modulation + ship_nb_noise

        return y, mean, logvar, ship_bb_noise, ship_bb_modulation, ship_nb_noise

    def shared_step(self, batch, stage: str):
        """
        Shared step for training and validation.

        Args:
            batch: input batch (B, 1, T)
            stage: "train" | "val"
        """
        x, _ = batch

        y, mean, logvar, _, _, _ = self.detailed_forward(x)

        recon = self.stft_loss(x, y)
        kl = DDSP_VAE._kl_loss(mean, logvar)

        loss = recon + self.hparams.beta_kl * kl

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}/recon", recon, on_epoch=True)
        self.log(f"{stage}/kl", kl, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")


    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
