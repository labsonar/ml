import typing
import torch
import torchaudio
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

            # log magnitude
            log_mag = torch.mean(torch.abs(torch.log(mag_X + 1e-7) - torch.log(mag_Y + 1e-7)))
            loss += log_mag

        return loss

class MultiLofarLoss(MultiSTFTLoss):

    @staticmethod
    def soft_tpsw(x, kernel_size=None, hole_size=None):
        # x: (B, F, T)

        B, F, T = x.shape
        device = x.device

        if kernel_size is None:
            kernel_size = int(round(F * .04 / 2.0 + 1))
            if kernel_size % 2 == 0:
                kernel_size += 1

        if hole_size is None:
            hole_size = int(round(kernel_size / 8.0 + 1))
            if hole_size % 2 == 0:
                hole_size += 1

        print("F: ", F)
        print("T: ", T)
        print("kernel_size: ", kernel_size)
        print("hole_size: ", hole_size)

        # cria kernel
        kernel = torch.ones(kernel_size, device=device)

        center = kernel_size // 2
        half_hole = hole_size // 2
        kernel[center - half_hole : center + half_hole + 1] = 0.0

        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)

        x_perm = x.permute(0, 2, 1)  # (B, T, F)
        x_reshaped = x_perm.reshape(B * T, 1, F)

        background = torch.nn.functional.conv1d(
            x_reshaped,
            kernel,
            padding=kernel_size // 2
        )

        # volta ao shape original
        background = background.reshape(B, T, F)
        background = background.permute(0, 2, 1)

        return torch.relu(torch.log(x/background))

    def stft(self, x, fft_size, hop):
        x = super().stft(x, fft_size, hop)
        x = torch.abs(x)
        # x = self.soft_tpsw(x)
        return x

class MultiMelLoss(torch.nn.Module):

    def __init__(
        self,
        sample_rate: int,
        n_ffts=[1024, 2048, 512],
        hop_sizes=[256, 512, 128],
        n_mels=80,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.transforms = torch.nn.ModuleList()

        for n_fft, hop in zip(n_ffts, hop_sizes):
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                power=1.0,  # magnitude (não potência)
                normalized=False,
            )
            self.transforms.append(mel)

    def forward(self, x, y):
        loss = 0.0

        for mel in self.transforms:
            X = mel(x.squeeze(1))  # (B, n_mels, T)
            Y = mel(y.squeeze(1))

            # spectral convergence
            sc = torch.norm(X - Y) / (torch.norm(X) + 1e-7)
            loss += sc

            # log-mel loss (recomendado)
            log_X = torch.log(X + 1e-7)
            log_Y = torch.log(Y + 1e-7)
            log_mag = torch.mean(torch.abs(log_X - log_Y))

            loss += log_mag

        return loss

class Decimate(torch.nn.Module):
    def __init__(self, factor, kernel_size=63):
        super().__init__()
        self.factor = factor

        # cria um low-pass sinc windowed
        t = torch.arange(kernel_size) - (kernel_size - 1) / 2
        sinc = torch.sinc(t / factor)

        window = torch.hann_window(kernel_size)
        kernel = sinc * window
        kernel = kernel / kernel.sum()

        self.register_buffer("kernel", kernel.view(1, 1, -1))

    def forward(self, x):
        x = torch.nn.functional.conv1d(
            x,
            self.kernel,
            stride=self.factor,
            padding=self.kernel.shape[-1] // 2
        )
        return x

class DemonLoss(torch.nn.Module):

    def __init__(
        self,
        sample_rate: float,
        n_fft: int = 512,
        hop_length: int = 256,
        decimate: typing.List[int] = [16, 8],
        eps: float = 1e-7,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.decimate = decimate
        self.eps = eps

        self.decimators = []
        for ratio in self.decimate:
            self.decimators.append(Decimate(ratio))

    def envelope(self, x):
        # return torch.sqrt(torch.nn.functional.avg_pool1d(x**2, kernel_size=32, stride=1, padding=16))
        return torch.abs(x)

    def decimate_signal(self, x: torch.Tensor):
        if self.decimate == 1:
            return x

        for decimator in self.decimators:
            x = decimator(x)
        # for ratio in self.decimate:
        #     x = torchaudio.functional.resample(x, orig_freq=self.sample_rate, new_freq=self.sample_rate//ratio)
        return x

    def demon_spectrogram(self, x: torch.Tensor):
        # x: (B, 1, T)

        print("decimated x: ", x.shape)

        x = self.envelope(x)
        x = self.decimate_signal(x)

        print("decimated x: ", x.shape)

        X = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            return_complex=True
        )

        X = torch.abs(X)

        return MultiLofarLoss.soft_tpsw(X)

    def forward(self, x, y):
        X = self.demon_spectrogram(x)
        Y = self.demon_spectrogram(y)

        # spectral convergence
        sc = torch.norm(X - Y) / (torch.norm(X) + self.eps)

        # log magnitude
        log_X = torch.log(X + self.eps)
        log_Y = torch.log(Y + self.eps)
        log_mag = torch.mean(torch.abs(log_X - log_Y))

        return sc + log_mag

class MultiDemonLoss(torch.nn.Module):

    def __init__(
        self,
        sample_rate,
        configs=[
            (512, 128, 5),
            (1024, 256, 10),
            (2048, 512, 20),
        ],
    ):
        super().__init__()

        self.losses = torch.nn.ModuleList([
            DemonLoss(sample_rate, n_fft, hop, dec)
            for (n_fft, hop, dec) in configs
        ])

    def forward(self, x, y):
        loss = 0.0
        for l in self.losses:
            loss += l(x, y)
        return loss

class Loss(torch.nn.Module):

    def __init__(self,
                 sample_rate: lps_qty.Frequency,
                 stft_factor = 1,
                 mel_factor = 1,
                 lofar_factor = 3):
        super().__init__()
        self.stft_factor = stft_factor
        self.mel_factor = mel_factor
        self.lofar_factor = lofar_factor
        self.stft_loss = MultiSTFTLoss()
        self.mel_loss = MultiMelLoss(sample_rate=int(sample_rate.get_hz()))
        self.lofar_loss = MultiLofarLoss()

    def forward(self, x, y):
        loss = 0
        loss += self.stft_factor * self.stft_loss(x, y)
        loss += self.mel_factor * self.mel_loss(x, y)
        loss += self.lofar_factor * self.lofar_loss(x, y)
        return loss

class DDSP_VAE(lightning.LightningModule):

    def __init__(
        self,

        n_bands: int = 8,
        capacity=32,
        latent_dim=64,

        kernel=3,
        ratios=[4, 4, 4, 2],

        noise_ratios=[8, 8, 4, 4], # 0,512s
        noise_bands=8,
        n_noise_channels=1,

        n_harmonics=8,
        sample_rate: lps_qty.Frequency = lps_qty.Frequency.khz(16),

        beta_kl=1e-1,
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

        bb_mod_ratios=[8, 8, 4, 4, 4] # 2s
        harmonic_head = ml_ddsp.BroadbandHarmonicModulatorHead(
            in_channels=dec_channels[-1],
            channels=[dec_channels[-1] for i in range(len(bb_mod_ratios))],
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

        nb_ratios=noise_ratios
        nb_head = ml_ddsp.NarrowbandHead(
            in_channels=dec_channels[-1],
            channels=[dec_channels[-1] for i in range(len(nb_ratios))],
            n_freqs=2,
            stride=nb_ratios,
        )

        self.nb = ml_ddsp.Narrowband(
            head=nb_head,
            sample_rate=sample_rate,
            samples_per_frame=n_bands,
            f_min = lps_qty.Frequency.hz(10.0),
            f_max = lps_qty.Frequency.hz(4000.0),

        )

        self.ch_ir = ml_ddsp.DifferentiableIR(
            duration = lps_qty.Time.s(1),
            sample_rate=sample_rate,
        )

        self.env = ml_ddsp.Broadband(
            in_channels=dec_channels[-1],
            hidden_size=dec_channels[-1],
            out_channels=n_bands,
            ratios=noise_ratios,
            noise_bands=noise_bands,
            n_noise_channels=n_noise_channels,
        )

        self.loss = Loss(sample_rate=sample_rate)

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

        ship = ship_bb_noise * ship_bb_modulation + ship_nb_noise

        signal = self.ch_ir(ship)
        print("signal: ", signal.shape)

        ir = self.ch_ir.build_impulse()
        print("ir: ", ir.shape)

        env_noise = self.env(features)
        print("env_noise: ", env_noise.shape)
        env_noise = self.pqmf.reverse(env_noise)
        print("env_noise pqmf: ", env_noise.shape)


        # y = signal + env_noise
        y = ship_bb_noise * ship_nb_noise
        print("y: ", y.shape)

        return y, mean, logvar, ship_bb_noise, ship_bb_modulation, ship_nb_noise, ship, ir, env_noise, signal

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
