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

class BroadbandHarmonicModulatorHead(ml_stack1d.ConvEncoder):

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
            project_dim=n_harmonics + 2,
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
        mod_index = params[:, 1:2]
        amps = params[:, 2:]

        return torch.sigmoid(f0), torch.sigmoid(mod_index), torch.relu(amps)

class BroadbandHarmonicModulator(torch.nn.Module):

    def __init__(
        self,
        head: BroadbandHarmonicModulatorHead,
        sample_rate: lps_qty.Frequency,
        samples_per_frame: int,
        f0_min: lps_qty.Frequency = lps_qty.Frequency.rpm(30),
        f0_max: lps_qty.Frequency = lps_qty.Frequency.rpm(300),
    ):
        super().__init__()

        self.head = head
        self.sample_rate = sample_rate.get_hz()
        self.samples_per_frame = samples_per_frame * head.compactness_factor

        self.n_harmonics = head.n_harmonics

        self.f0_min = f0_min.get_hz()
        self.f0_max = f0_max.get_hz()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T_frames)

        Returns:
            signal: (B, 1, T_audio)
        """

        print(f"[BBHarmonicModulator] input: {x.shape}")

        f0, mod_index, amps = self.head(x)

        print(f"\t[BBHarmonicModulator] f0: {f0.shape}")
        print(f"\t[BBHarmonicModulator] amps: {amps.shape}")

        f0 = self.f0_min + (self.f0_max - self.f0_min) * f0
        print(f"\t[BBHarmonicModulator] f0: {f0.shape} -> {f0.min().item()*60:.1f} rpm to {f0.max().item()*60:.1f} rpm")

        a0 = torch.sum(amps, dim=1, keepdim=True)/ (mod_index + 1e-6)
        total_energy = a0**2 + torch.sum(amps**2, dim=1, keepdim=True)/ 2

        a0 = a0 / (torch.sqrt(total_energy) + 1e-6)
        amps = amps / (torch.sqrt(total_energy) + 1e-6)

        print(f"\t[BBHarmonicModulator] f0: {f0}")
        print(f"\t[BBHarmonicModulator] a0: {a0}")
        print(f"\t[BBHarmonicModulator] amps: {amps}")

        # n_samples = self.samples_per_frame * f0.shape[-1]
        # f0 = torch.nn.functional.interpolate(f0,
        #             size=n_samples, mode='linear', align_corners=True)
        # a0 = torch.nn.functional.interpolate(a0,
        #             size=n_samples, mode='linear', align_corners=True)
        # amps = torch.nn.functional.interpolate(amps,
        #             size=n_samples, mode='linear', align_corners=True)

        f0 = f0.repeat_interleave(self.samples_per_frame, dim=2)
        a0 = a0.repeat_interleave(self.samples_per_frame, dim=2)
        amps = amps.repeat_interleave(self.samples_per_frame, dim=2)
        print(f"\t[BBHarmonicModulator] repeat f0: {f0.shape}")
        print(f"\t[BBHarmonicModulator] repeat a0: {a0.shape}")
        print(f"\t[BBHarmonicModulator] repeat amps: {amps.shape}")

        omega = 2 * torch.pi * f0 / self.sample_rate
        print(f"\t[BBHarmonicModulator] omega: {omega.shape}")
        phase = torch.cumsum(omega, dim=-1)
        print(f"\t[BBHarmonicModulator] phase: {phase.shape}")

        k = torch.arange(
            1, self.n_harmonics + 1,
            device=x.device
        ).view(1, -1, 1)
        print(f"\t[BBHarmonicModulator] k: {k.shape}")

        harm = torch.sum(
            amps * torch.sin(k * phase),
            dim=1,
            keepdim=True
        )
        print(f"\t[BBHarmonicModulator] signal: {harm.shape}")

        signal = a0 + harm

        return signal

class NarrowbandHead(ml_stack1d.ConvEncoder):

    def __init__(
        self,
        in_channels: int,
        channels: typing.List[int],
        n_freqs: int,

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
            project_dim=2*n_freqs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )

        self.n_freqs = n_freqs

    def forward(self, x: torch.Tensor):
        params = super().forward(x)
        freqs, amps = torch.chunk(params, 2, dim=1)
        return torch.sigmoid(freqs), torch.relu(amps)

class Narrowband(torch.nn.Module):

    def __init__(
        self,
        head: NarrowbandHead,
        sample_rate: lps_qty.Frequency,
        samples_per_frame: int,
        f_min: lps_qty.Frequency = lps_qty.Frequency.hz(10.0),
        f_max: lps_qty.Frequency | None = None,
    ):
        super().__init__()

        self.head = head
        self.sample_rate = sample_rate.get_hz()
        self.samples_per_frame = samples_per_frame * head.compactness_factor

        self.n_harmonics = head.n_freqs

        self.f_min = f_min.get_hz()
        self.f_max = (f_max or (sample_rate / 2.0)).get_hz()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T_frames)

        Returns:
            signal: (B, 1, T_audio)
        """

        print(f"[Narrowband] input: {x.shape}")

        freqs, amps = self.head(x)

        print(f"\t[Narrowband] freqs: {freqs.shape}")
        print(f"\t[Narrowband] amps: {amps.shape}")

        freqs = self.f_min + (self.f_max - self.f_min) * torch.sigmoid(freqs)
        print(f"\t[Narrowband] freqs: {freqs.shape} -> {freqs.min().item():.2f} hz to {freqs.max().item():.2f} hz")
        print(f"\t[Narrowband] amps: {amps.shape} -> {amps.min().item():.2f} to {amps.max().item():.2f}")

        # n_samples = self.samples_per_frame * f0.shape[-1]
        # f0 = torch.nn.functional.interpolate(f0,
        #             size=n_samples, mode='linear', align_corners=True)
        # a0 = torch.nn.functional.interpolate(a0,
        #             size=n_samples, mode='linear', align_corners=True)
        # amps = torch.nn.functional.interpolate(amps,
        #             size=n_samples, mode='linear', align_corners=True)

        freqs = freqs.repeat_interleave(self.samples_per_frame, dim=2)
        amps = amps.repeat_interleave(self.samples_per_frame, dim=2)
        print(f"\t[Narrowband] repeat freqs: {freqs.shape}")
        print(f"\t[Narrowband] repeat amps: {amps.shape}")

        omega = 2 * torch.pi * freqs / self.sample_rate
        print(f"\t[Narrowband] omega: {omega.shape}")

        phase = torch.cumsum(omega, dim=-1)
        print(f"\t[Narrowband] phase: {phase.shape}")

        mask = (freqs <= self.sample_rate / 2).float()
        print(f"\t[Narrowband] mask: {mask.shape}")

        signal = torch.sum(
            amps * mask * torch.sin(phase),
            dim=1,
            keepdim=True
        )
        print(f"\t[Narrowband] signal: {signal.shape}")

        return signal
