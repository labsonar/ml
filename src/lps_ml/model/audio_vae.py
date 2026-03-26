import typing
import torch
import torchaudio
import lightning

import lps_utils.quantities as lps_qty
import lps_ml.model.blocks.stack1d as ml_stack1d
import lps_ml.utils.pqmf as ml_pqmf
import lps_ml.model.blocks.ddsp as ml_ddsp
import lps_ml.utils.sonar_loss as ml_loss

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

class DDSP_VAE(lightning.LightningModule):

    def __init__(
        self,

        n_bands: int = 8,
        capacity=16,
        latent_dim=32,

        kernel=3,
        ratios=[4, 4, 4, 2],

        noise_ratios=[8, 8, 4, 4], # 0,512s
        bb_mod_ratios=[8, 8, 4, 4, 4], # 2s
        nb_ratios=[4, 4, 4, 4], # 0,512s
        noise_bands=8,
        n_noise_channels=1,

        n_harmonics=8,
        sample_rate: lps_qty.Frequency = lps_qty.Frequency.khz(16),

        beta_kl=0.1,
        stft_factor=1,
        mel_factor=0,
        lofar_factor=0,
        demon_factor=0,

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
            in_channels=capacity,
            hidden_size=capacity,
            out_channels=n_bands,
            ratios=noise_ratios,
            noise_bands=noise_bands,
            n_noise_channels=n_noise_channels,
        )

        harmonic_head = ml_ddsp.BroadbandHarmonicModulatorHead(
            in_channels=capacity,
            channels=[capacity for i in range(len(bb_mod_ratios))],
            n_harmonics=n_harmonics,
            stride=bb_mod_ratios,
        )

        self.bb_mod = ml_ddsp.BroadbandHarmonicModulator(
            head=harmonic_head,
            sample_rate=sample_rate,
            samples_per_frame=n_bands,
            f0_min=lps_qty.Frequency.rpm(40),
            f0_max=lps_qty.Frequency.rpm(200),
        )

        nb_head = ml_ddsp.NarrowbandHead(
            in_channels=capacity,
            channels=[capacity for i in range(len(nb_ratios))],
            n_freqs=1,
            kernel_size=3,
            stride=nb_ratios,
            dilation=[1 + (2*i) for i in range(len(nb_ratios))],
        )

        self.nb = ml_ddsp.Narrowband(
            head=nb_head,
            sample_rate=sample_rate,
            samples_per_frame=n_bands,
            f_min = lps_qty.Frequency.hz(10),
            f_max = lps_qty.Frequency.khz(4),

        )

        nb_harm_head = ml_ddsp.NarrowbandHarmonicHead(
            in_channels=capacity,
            # channels=[capacity for i in range(len(nb_ratios))],
            n_harmonics=12,
            # stride=nb_ratios,
            # dilation=[1 + (2*i) for i in range(len(nb_ratios))],
            channels = [2, 4, 8, 16, 16, 32, 64, 64],
            stride  = [4, 4, 4, 4, 4, 4, 4, 4],
            kernel_size  = [7, 7, 7, 7, 7, 7, 7, 7],
            dilation=[7, 7 , 5, 5, 3, 3, 1, 1],
        )

        self.nb_harm = ml_ddsp.NarrowbandHarmonic(
            head=nb_harm_head,
            sample_rate=sample_rate,
            samples_per_frame=n_bands,
            f_min = lps_qty.Frequency.hz(10.0),
            f_max = lps_qty.Frequency.khz(8),
        )

        # nb_harm_head = ml_ddsp.NarrowbandHead(
        #     in_channels=capacity,
        #     # channels=[capacity for i in range(len(nb_ratios))],
        #     n_freqs=6,
        #     # stride=nb_ratios,
        #     # dilation=[1 + (2*i) for i in range(len(nb_ratios))],
        #     channels = [8, 16, 32, 64, 128],
        #     stride  = [8, 8, 8, 4, 4],
        #     kernel_size  = [11, 11, 9, 7, 5]
        # )

        # self.nb_harm = ml_ddsp.Narrowband(
        #     head=nb_harm_head,
        #     sample_rate=sample_rate,
        #     samples_per_frame=n_bands,
        #     f_min = lps_qty.Frequency.hz(10.0),
        #     f_max = lps_qty.Frequency.khz(4),
        # )

        self.ch_ir = ml_ddsp.DifferentiableIR(
            duration = lps_qty.Time.s(1),
            sample_rate=sample_rate,
        )

        self.env = ml_ddsp.Broadband(
            in_channels=capacity,
            hidden_size=capacity,
            out_channels=n_bands,
            ratios=noise_ratios,
            noise_bands=noise_bands,
            n_noise_channels=n_noise_channels,
        )

        self.loss = ml_loss.SonarLoss(
            stft_factor=stft_factor,
            mel_factor=mel_factor,
            lofar_factor=lofar_factor,
            demon_factor=demon_factor,
        )

    @staticmethod
    def _reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    @staticmethod
    def _kl_loss(mean, logvar):
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    def forward(self, x):
        y, _, _, _, _, _, _, _, _, _ = self.detailed_forward(x)
        return y

    def detailed_forward(self, x):
        """
        x: (B, 1, T)
        """
        #print("input (B, 1, T): ", x.shape)

        if self.hparams.n_bands > 1:
            x_sub = self.pqmf(x)
        else:
            x_sub = x
        #print("x_sub: ", x_sub.shape)

        # mean, logvar = self.encoder(x_sub)
        #print("mean: ", mean.shape)
        #print("logvar: ", logvar.shape)

        # z = DDSP_VAE._reparameterize(mean, logvar)
        #print("z: ", z.shape)

        # features = self.decoder(z)
        #print("features: ", features.shape)

        # ship_bb_noise = self.bb(x_sub)
        # # #print("ship_bb_noise: ", ship_bb_noise.shape)
        # if self.hparams.n_bands > 1:
        #     ship_bb_noise = self.pqmf.reverse(ship_bb_noise)
        # #print("ship_bb_noise pqmf: ", ship_bb_noise.shape)

        # ship_bb_modulation = self.bb_mod(features)
        # #print("ship_bb_modulation: ", ship_bb_modulation.shape)

        # ship_nb_noise = self.nb(features)
        # #print("ship_nb_noise: ", ship_nb_noise.shape)

        # ship = ship_bb_noise * ship_bb_modulation + ship_nb_noise

        # signal = self.ch_ir(ship)
        # #print("signal: ", signal.shape)

        # ir = self.ch_ir.build_impulse()
        # #print("ir: ", ir.shape)

        # env_noise = self.env(features)
        # #print("env_noise: ", env_noise.shape)
        # env_noise = self.pqmf.reverse(env_noise)
        # #print("env_noise pqmf: ", env_noise.shape)

        nb = self.nb_harm(x_sub)

        # y = signal + env_noise
        y = nb
        #print("y: ", y.shape)

        return y, None, None, None, None, None, None, None, None, None
        # return y, mean, logvar, ship_bb_noise, ship_bb_modulation, ship_nb_noise, ship, ir, env_noise, signal

    def shared_step(self, batch, stage: str):
        """
        Shared step for training and validation.

        Args:
            batch: input batch (B, 1, T)
            stage: "train" | "val"
        """
        x, _ = batch

        y, mean, logvar, _, _, _, _, _, _, _ = self.detailed_forward(x)

        recon = self.loss(x, y)
        # kl = DDSP_VAE._kl_loss(mean, logvar)

        loss = recon
        # loss = recon + self.hparams.beta_kl * kl

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}/recon", recon, on_epoch=True)
        # self.log(f"{stage}/kl", kl, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
