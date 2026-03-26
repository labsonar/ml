import abc
import typing
import dataclasses

import lps_utils.quantities as lps_qty

import torch
import torchaudio

class AudioProcessor(torch.nn.Module, abc.ABC):
    """
    Classe base para transformar sinal [..., T] -> [..., F, T]
    ou [..., F] se temporal_mean=True
    """

    def __init__(self,
                 temporal_mean: bool = False,
                 temporal_integration: int | None = None):
        super().__init__()
        self.temporal_mean = temporal_mean
        self.temporal_integration = temporal_integration

    @staticmethod
    def soft_tpsw_norm(
        x: torch.Tensor,
        freq_dim: int = -2,
        kernel_size: int = None,
        hole_size: int = None,
    ):

        freq_dim = freq_dim % x.ndim
        n_freqs = x.shape[freq_dim]

        if kernel_size is None:
            kernel_size = int(round(n_freqs * 0.04 / 2.0 + 1))
            if kernel_size % 2 == 0:
                kernel_size += 1

        if hole_size is None:
            hole_size = int(round(kernel_size / 8.0 + 1))
            if hole_size % 2 == 0:
                hole_size += 1

        device = x.device

        kernel = torch.ones(kernel_size, device=device)

        center = kernel_size // 2
        half_hole = hole_size // 2
        kernel[center - half_hole:center + half_hole + 1] = 0.0

        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, -1)

        x_perm = x.transpose(freq_dim, -1)  # [..., T, F]
        orig_shape = x_perm.shape
        F_len = orig_shape[-1]

        x_flat = x_perm.reshape(-1, 1, F_len)

        background = torch.nn.functional.conv1d(
            x_flat,
            kernel,
            padding=kernel_size // 2
        )

        background = background.reshape(orig_shape)
        background = background.transpose(freq_dim, -1)
        return torch.relu(torch.log(x) - torch.log(background))

    @abc.abstractmethod
    def process(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retorna [..., F, T]
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.process(x)

        y = y[..., 1:, :]

        if self.temporal_integration:
            y = self.apply_temporal_integration(y)

        if self.temporal_mean:
            return y.mean(dim=-1)

        return y

    def apply_temporal_integration(self, y: torch.Tensor) -> torch.Tensor:

        k = self.temporal_integration

        n_samples = y.shape[-1]

        if k >= n_samples:
            if self.temporal_mean:
                return y
            return y.mean(dim=-1)

        T_trim = (n_samples // k) * k
        y = y[..., :T_trim]

        new_shape = (*y.shape[:-1], T_trim // k, k)
        y = y.view(new_shape)

        return y.mean(dim=-1)

@dataclasses.dataclass
class STFTConfig:
    n_fft: int = 1024
    hop_length: int = 512
    temporal_mean: bool = False
    temporal_integration: int | None = None

class STFT(AudioProcessor):

    def __init__(self,
                 stft_config: STFTConfig):
        super().__init__(stft_config.temporal_mean, stft_config.temporal_integration)
        self.stft_config = stft_config

    def process(self, x):
        ret = torch.stft(
            x.squeeze(1),
            n_fft=self.stft_config.n_fft,
            hop_length=self.stft_config.hop_length,
            win_length=self.stft_config.n_fft,
            return_complex=True
        )
        return torch.abs(ret)

@dataclasses.dataclass
class MelConfig(STFTConfig):
    sample_rate: lps_qty.Frequency = lps_qty.Frequency.khz(16)
    n_mels: int = 80
    f_min: lps_qty.Frequency = lps_qty.Frequency.hz(0)
    f_max: lps_qty.Frequency | None = None

class Mel(AudioProcessor):

    def __init__(
        self,
        mel_config: MelConfig,
    ):
        super().__init__(mel_config.temporal_mean, mel_config.temporal_integration)

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(mel_config.sample_rate.get_hz()),
            n_fft=mel_config.n_fft,
            hop_length=mel_config.hop_length,
            win_length=mel_config.n_fft,
            n_mels=mel_config.n_mels,
            f_min=int(mel_config.f_min.get_hz()),
            f_max=int(mel_config.f_max.get_hz()) if mel_config.f_max is not None else None,
            power=1.0,
        )

    def process(self, x):
        return self.mel(x.squeeze(1))

LofarConfig = STFTConfig
class Lofar(STFT):

    def process(self, x):
        y = super().process(x)
        return AudioProcessor.soft_tpsw_norm(y)

class Decimate(torch.nn.Module):
    def __init__(self,
                 factor: int,
                 kernel_size: int = 63):
        super().__init__()
        self.factor = factor

        t = torch.arange(kernel_size) - (kernel_size - 1) / 2
        sinc = torch.sinc(t / factor)
        window = torch.hann_window(kernel_size)

        kernel = (sinc * window)
        kernel = kernel / kernel.sum()

        self.register_buffer("kernel", kernel.view(1, 1, -1))

    def forward(self, x):
        return torch.nn.functional.conv1d(
            x,
            self.kernel,
            stride=self.factor,
            padding=self.kernel.shape[-1] // 2
        )

@dataclasses.dataclass
class DemonConfig(STFTConfig):
    decimate: typing.List[int] = dataclasses.field(default_factory=lambda: [8, 8])

class Demon(AudioProcessor):

    def __init__(
        self,
        demon_config = DemonConfig,
    ):
        super().__init__(demon_config.temporal_mean, demon_config.temporal_integration)

        self.demon_config = demon_config

        self.decimators = torch.nn.ModuleList(
            [Decimate(d) for d in demon_config.decimate]
        )

    def process(self, x):
        x = torch.abs(x)

        for d in self.decimators:
            x = d(x)

        y = torch.stft(
            x.squeeze(1),
            n_fft=self.demon_config.n_fft,
            hop_length=self.demon_config.hop_length,
            win_length=self.demon_config.n_fft,
            return_complex=True
        )

        y = torch.abs(y)

        return AudioProcessor.soft_tpsw_norm(y)

class MultiResolutionLoss(torch.nn.Module):
    """
    Classe base (não usada diretamente).
    Use:
        MultiResolutionLoss[Processor](configs)
    """

    def __init__(self,
                 processor_cls: typing.Type[AudioProcessor],
                 configs: typing.List[typing.Union[dict, object]],
                 compute_log: bool = False,
                 eps: float = 1e-7):
        super().__init__()

        self.processors = torch.nn.ModuleList([
            processor_cls(cfg)
            for cfg in configs
        ])
        self.eps = eps
        self.compute_log = compute_log

    def forward(self, input_a, input_b):
        loss = 0.0

        for proc in self.processors:
            proc_a = proc(input_a)
            proc_b = proc(input_b)

            sc = torch.norm(proc_a - proc_b) / (torch.norm(proc_a) + self.eps)
            loss += sc

            if self.compute_log:
                log_a = torch.log(proc_a + self.eps)
                log_b = torch.log(proc_b + self.eps)

                log_mag = torch.mean(torch.abs(log_a - log_b))

                loss += log_mag

        return loss

    def __class_getitem__(cls, processor_cls: typing.Type[AudioProcessor]):

        class _TypedMultiResolutionLoss(cls):
            def __init__(self, configs, compute_log = False, eps = 1e-7):
                super().__init__(processor_cls, configs, compute_log, eps)

        _TypedMultiResolutionLoss.__name__ = f"{cls.__name__}[{processor_cls.__name__}]"

        return _TypedMultiResolutionLoss

class SonarLoss(torch.nn.Module):

    def __init__(
        self,
        stft_factor=1.0,
        stft_loss: MultiResolutionLoss | None = None,

        mel_factor=1.0,
        mel_loss: MultiResolutionLoss | None = None,

        lofar_factor=1.0,
        lofar_loss: MultiResolutionLoss | None = None,

        demon_factor=1.0,
        demon_loss: MultiResolutionLoss | None = None,
    ):
        super().__init__()

        self.stft_factor = stft_factor
        self.mel_factor = mel_factor
        self.lofar_factor = lofar_factor
        self.demon_factor = demon_factor

        self.stft_loss = stft_loss or MultiResolutionLoss[STFT]([
            STFTConfig(256, 128, temporal_integration=30),
            STFTConfig(1024, 512, temporal_integration=20),
            STFTConfig(4096, 2048, temporal_integration=10),
        ])

        self.mel_loss = mel_loss or MultiResolutionLoss[Mel]([
            MelConfig(256, 128, n_mels=64, temporal_integration=30),
            MelConfig(1024, 512, n_mels=128, temporal_integration=20),
            MelConfig(4096, 2048, n_mels=256, temporal_integration=10),
        ])

        self.lofar_loss = lofar_loss or MultiResolutionLoss[Lofar]([
            LofarConfig(256, 128, temporal_integration=30),
            LofarConfig(1024, 512, temporal_integration=20),
            LofarConfig(4096, 2048,temporal_integration=10),
        ])

        self.demon_loss = demon_loss or MultiResolutionLoss[Demon]([
            DemonConfig(256, 128, temporal_integration=5, decimate=[32, 16]),
            DemonConfig(512, 256, temporal_integration=10, decimate=[16, 16]),
            DemonConfig(1024, 512, temporal_integration=20, decimate=[16, 8]),
        ])

    def forward(self, x, y):
        loss = 0.0

        if self.stft_factor:
            loss += self.stft_factor * self.stft_loss(x, y)

        if self.mel_factor:
            loss += self.mel_factor * self.mel_loss(x, y)

        if self.lofar_factor:
            loss += self.lofar_factor * self.lofar_loss(x, y)

        if self.demon_factor:
            loss += self.demon_factor * self.demon_loss(x, y)

        return loss
