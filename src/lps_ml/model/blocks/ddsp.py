import typing
import torch

import lps_utils.quantities as lps_qty
import lps_ml.model.blocks.stack1d as ml_stack1d

class Broadband(torch.nn.Module):
    """
    Broadband noise generator (DDSP-style) inspired by RAVE.

    This module learns a time-varying spectral envelope, converts it to an
    impulse response, and applies it to white noise via FFT convolution.

    The implementation is adapted from NoiseGenerator on RAVE:
    https://github.com/acids-ircam/RAVE/blob/master/rave/blocks.py

    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        ratios: typing.List[int],
        noise_bands: int,
        n_noise_channels: int = 1,
        activation: typing.Callable[[int], torch.nn.Module] | None = None,
    ):
        super().__init__()

        activation = activation or (lambda dim: torch.nn.LeakyReLU(0.2))

        self.out_channels = out_channels          # subbands (PQMF bands)
        self.noise_bands = noise_bands            # freq resolution
        self.n_noise_channels = n_noise_channels  # number of noise sources per band

        channels = [in_channels]
        channels += [hidden_size] * (len(ratios) - 1)
        channels += [out_channels * noise_bands * n_noise_channels]

        self.net = ml_stack1d.ConvEncoder(
            in_channels=in_channels,
            channels=[hidden_size] * len(ratios),
            project_dim=out_channels * noise_bands * n_noise_channels,
            kernel_size=[2 * r for r in ratios],
            stride=ratios,
            activation=lambda: torch.nn.LeakyReLU(0.2),
            norm=None,
        )

        # layers = []
        # for i, r in enumerate(ratios):
        #     layers.append(
        #         torch.nn.Conv1d(
        #             channels[i],
        #             channels[i + 1],
        #             kernel_size=2 * r,
        #             stride=r,
        #             padding=r//2,
        #         )
        #     )
        #     if i != len(ratios) - 1:
        #         layers.append(activation(channels[i + 1]))

        # self.net = torch.nn.Sequential(*layers)

        self.target_size = int(torch.prod(torch.tensor(ratios)).item())

    @staticmethod
    def amp_to_impulse_response(
        amp: torch.Tensor,
        target_size: int,
    ) -> torch.Tensor:
        """
        Convert spectral amplitude to impulse response.

        Adapted from:
        https://github.com/acids-ircam/RAVE

        Args:
            amp: (..., freq_bins)
            target_size: desired IR size

        Returns:
            Impulse response tensor (..., target_size)
        """

        amp = torch.stack([amp, torch.zeros_like(amp)], -1)
        amp = torch.view_as_complex(amp)
        amp = torch.fft.irfft(amp)

        filter_size = amp.shape[-1]

        amp = torch.roll(amp, filter_size // 2, -1)

        win = torch.hann_window(
            filter_size,
            dtype=amp.dtype,
            device=amp.device,
        )

        amp = amp * win

        amp = torch.nn.functional.pad(
            amp,
            (0, int(target_size) - int(filter_size)),
        )

        amp = torch.roll(amp, -filter_size // 2, -1)

        return amp

    @staticmethod
    def fft_convolve(
        signal: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """
        FFT-based convolution.

        Adapted from:
        https://github.com/acids-ircam/RAVE

        Args:
            signal: input signal
            kernel: convolution kernel (impulse response)

        Returns:
            Convolved signal
        """

        signal = torch.nn.functional.pad(signal, (0, signal.shape[-1]))
        kernel = torch.nn.functional.pad(kernel, (kernel.shape[-1], 0))

        output = torch.fft.irfft(
            torch.fft.rfft(signal) * torch.fft.rfft(kernel)
        )

        output = output[..., output.shape[-1] // 2:]

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)

        Returns:
            noise: (B, out_channels, T_out)
        """

        print("\t[BB] input: ", x.shape)

        # for i, layer in enumerate(self.net):
        #     x = layer(x)
        #     print(f"\t[BB] Layer {i} ({layer.__class__.__name__}): {x.shape}")
        # amp = torch.sigmoid(x - 5) # as done in RAVE

        amp = torch.sigmoid(self.net(x) - 5) # as done in RAVE
        print("\t[BB] sigmoid: ", amp.shape)

        # (B, C, T) → (B, T, C)
        amp = amp.permute(0, 2, 1)
        print("\t[BB] permute: ", amp.shape)

        # reshape:
        # (B, T, subbands * noise_channels, freq_bins)
        amp = amp.reshape(
            amp.shape[0],
            amp.shape[1],
            self.out_channels * self.n_noise_channels,
            self.noise_bands,
        )
        print("\t[BB] reshape: ", amp.shape)

        ir = self.amp_to_impulse_response(
            amp,
            self.target_size,
        )
        print("\t[BB] ir: ", ir.shape)
        noise = torch.rand_like(ir) * 2 - 1
        print("\t[BB] noise: ", noise.shape)

        noise = self.fft_convolve(noise, ir)
        print("\t[BB] fft_convolve: ", noise.shape)

        # (B, T, subbands * noise_channels, time)
        noise = noise.permute(0, 2, 1, 3)
        print("\t[BB] permute: ", noise.shape)

        noise = noise.reshape(
            noise.shape[0],
            self.out_channels,
            self.n_noise_channels,
            -1,
        )
        print("\t[BB] reshape: ", noise.shape)

        noise = noise.sum(dim=2)
        print("\t[BB] sum: ", noise.shape)

        return noise

class HarmonicHead(ml_stack1d.ConvEncoder):
    """
    ConvEncoder specialized for harmonic parameter estimation.

    Output channels:
        channel 0      → f0
        channels 1..N  → harmonic amplitudes
    """

    def __init__(
        self,
        in_channels: int,
        channels: typing.List[int],
        n_harmonics: int,

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
            project_dim=n_harmonics + 1,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )

        self.n_harmonics = n_harmonics

    def forward(self, x: torch.Tensor):
        params = super().forward(x)

        f0 = params[:, :1]
        amps = params[:, 1:]

        return f0, amps

class Harmonic(torch.nn.Module):
    """
    DDSP-style harmonic synthesizer.

    Uses:
        HarmonicHead → parameter estimation
        Oscillator bank → signal synthesis
    """

    def __init__(
        self,
        head: HarmonicHead,
        sample_rate: lps_qty.Frequency,
        samples_per_frame: int,
        f0_min: lps_qty.Frequency = lps_qty.Frequency.hz(10.0),
        f0_max: lps_qty.Frequency | None = None,
    ):
        super().__init__()

        self.head = head
        self.sample_rate = sample_rate.get_hz()
        self.samples_per_frame = samples_per_frame * head.compactness_factor

        self.n_harmonics = head.n_harmonics

        self.f0_min = f0_min.get_hz()
        self.f0_max = (f0_max or (sample_rate / 2.0)).get_hz()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T_frames)

        Returns:
            signal: (B, 1, T_audio)
        """

        print(f"[Harmonic] input: {x.shape}")

        f0, amps = self.head(x)

        print(f"\t[Harmonic] f0: {f0.shape}")
        print(f"\t[Harmonic] amps: {amps.shape}")

        f0 = self.f0_min + (self.f0_max - self.f0_min) * torch.sigmoid(f0)
        print(f"\t[Harmonic] f0: {f0.shape} -> {f0.min().item()*60:.1f} rpm to {f0.max().item()*60:.1f} rpm")

        amps = torch.relu(amps)
        print(f"\t[Harmonic] relu: {amps.shape}")

        f0 = f0.repeat_interleave(self.samples_per_frame, dim=2)
        amps = amps.repeat_interleave(self.samples_per_frame, dim=2)
        print(f"\t[Harmonic] repeat f0: {f0.shape}")
        print(f"\t[Harmonic] repeat amps: {amps.shape}")

        omega = 2 * torch.pi * f0 / self.sample_rate
        print(f"\t[Harmonic] omega: {omega.shape}")
        phase = torch.cumsum(omega, dim=-1)
        print(f"\t[Harmonic] phase: {phase.shape}")

        k = torch.arange(
            1, self.n_harmonics + 1,
            device=x.device
        ).view(1, -1, 1)
        print(f"\t[Harmonic] k: {k.shape}")

        harmonic_freqs = k * f0
        print(f"\t[Harmonic] k: {harmonic_freqs.shape}")
        mask = (harmonic_freqs <= self.f0_max).float()

        signal = torch.sum(
            amps * mask * torch.sin(k * phase),
            dim=1,
            keepdim=True
        )
        print(f"\t[Harmonic] signal: {signal.shape}")

        return signal


