import typing
import math
import torch

import lps_ml.model.blocks.conv1d as lps_conv1d


def _expand_param(value, n):
    """Expand scalar parameter to list."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(
                f"Parameter list must have length {n}, got {len(value)}"
            )
        return value
    return [value] * n

class ConvFeatureExtractor(torch.nn.Module):
    """
    Stack of Conv1DLayer blocks forming an encoder.

    Parameters may be given either as a single value (applied to all layers)
    or as a list matching the number of layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: typing.List[int],

        kernel_size: typing.Union[int, typing.List[int]] = 7,
        stride: typing.Union[int, typing.List[int]] = 2,
        dilation: typing.Union[int, typing.List[int]] = 1,

        activation: typing.Callable = torch.nn.LeakyReLU,
        norm: typing.Optional[typing.Callable] = torch.nn.BatchNorm1d,
        dropout: float = 0.0,
    ):
        super().__init__()

        n_layers = len(out_channels)

        kernels = _expand_param(kernel_size, n_layers)
        strides = _expand_param(stride, n_layers)
        dilations = _expand_param(dilation, n_layers)

        channels = [in_channels] + list(out_channels)

        layers = []

        for i in range(n_layers):

            layers.append(
                lps_conv1d.Conv1DBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    dilation=dilations[i],
                    activation=activation,
                    norm=norm,
                    dropout=dropout,
                )
            )

        self.encoder = torch.nn.Sequential(*layers)

        self.compactness_factor = math.prod(strides)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """Forward pass through encoder."""
    #     return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"\t[Encoder] Input: {x.shape}")

        for i, layer in enumerate(self.encoder):
            x = layer(x)
            print(f"\t[Encoder] Layer {i} ({layer.__class__.__name__}): {x.shape}")

        return x

class ConvEncoder(torch.nn.Module):
    """
    Variational convolutional encoder.

    The encoder consists of:
        Conv1DFeatureExtractor -> Conv1d projection
    """

    def __init__(
        self,
        in_channels: int,
        channels: typing.List[int],
        project_dim: int,

        kernel_size: typing.Union[int, typing.List[int]] = 7,
        stride: typing.Union[int, typing.List[int]] = 2,
        dilation: typing.Union[int, typing.List[int]] = 1,

        activation: typing.Callable = torch.nn.LeakyReLU,
        norm: typing.Optional[typing.Callable] = torch.nn.BatchNorm1d,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.feature_extractor = ConvFeatureExtractor(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            activation=activation,
            norm=norm,
            dropout=dropout,
        )

        last_channels = channels[-1]

        self.latent_projection = torch.nn.Conv1d(
            in_channels=last_channels,
            out_channels=project_dim,
            kernel_size=1
        )

        self.project_dim = project_dim

        self.compactness_factor = self.feature_extractor.compactness_factor

    def forward(self, x: torch.Tensor):
        features = self.feature_extractor(x)
        project_params = self.latent_projection(features)
        print(f"\t[Encoder] latent_projection: {project_params.shape}")
        return project_params

class ResidualStack(torch.nn.Module):
    """
    Stack of ResidualBlock1D.

    Parameters may be scalar or list (per layer).
    """

    def __init__(
        self,
        channels: int,
        n_blocks: int,

        kernel_size: typing.Union[int, typing.List[int]] = 3,
        dilation: typing.Union[int, typing.List[int]] = 1,
        activation: typing.Callable = torch.nn.LeakyReLU,
    ):
        super().__init__()

        kernels = self._expand_param(kernel_size, n_blocks)
        dilations = self._expand_param(dilation, n_blocks)

        layers = []

        for i in range(n_blocks):
            layers.append(
                lps_conv1d.ResidualBlock(
                    channels=channels,
                    kernel_size=kernels[i],
                    dilation=dilations[i],
                    activation=activation,
                )
            )

        self.stack = torch.nn.Sequential(*layers)

    @staticmethod
    def _expand_param(value, n):
        if isinstance(value, (list, tuple)):
            if len(value) != n:
                raise ValueError(
                    f"Parameter list must have length {n}, got {len(value)}"
                )
            return value
        return [value] * n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)

class Upsampler(torch.nn.Module):
    """
        Upsampling1D -> ResidualStack1D
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,

        # upsampling
        up_factor: int = 2,
        activation: typing.Callable = torch.nn.LeakyReLU,

        # residual stack
        n_residual_blocks: int = 3,
        residual_kernel_size: typing.Union[int, typing.List[int]] = 3,
        residual_dilation: typing.Union[int, typing.List[int]] = 1,
    ):
        super().__init__()

        self.upsample = lps_conv1d.UpsamplingBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2*up_factor,
            stride=up_factor,
            padding=up_factor//2,
            activation=activation,
        )

        self.res_stack = ResidualStack(
            channels=out_channels,
            n_blocks=n_residual_blocks,
            kernel_size=residual_kernel_size,
            dilation=residual_dilation,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.res_stack(x)
        return x

class ConvFeatureReconstructor(torch.nn.Module):
    """
    Stack of Upsampler.

    Each block increases temporal resolution.
    """

    def __init__(
        self,
        in_channels: int,
        channels: typing.List[int],

        adapt_channels: int | None = None,

        # upsampling params
        up_factors: typing.Union[int, typing.List[int]] = 2,

        # residual params
        n_res_blocks: typing.Union[int, typing.List[int]] = 3,
        res_kernel_size: typing.Union[int, typing.List[int]] = 3,
        res_dilation: typing.Union[int, typing.List[int]] = 1,

        activation: typing.Callable = torch.nn.LeakyReLU,
    ):
        super().__init__()

        n_layers = len(channels)

        up_factors = _expand_param(up_factors, n_layers)
        n_res_blocks = _expand_param(n_res_blocks, n_layers)

        # res params podem ser compartilhados ou por camada
        if isinstance(res_kernel_size, (list, tuple)) and isinstance(res_kernel_size[0], (list, tuple)):
            res_kernel_sizes = res_kernel_size
        else:
            res_kernel_sizes = [res_kernel_size] * n_layers

        if isinstance(res_dilation, (list, tuple)) and isinstance(res_dilation[0], (list, tuple)):
            res_dilations = res_dilation
        else:
            res_dilations = [res_dilation] * n_layers

        layers = []

        if adapt_channels is not None:
            layers.append(
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=adapt_channels,
                    kernel_size=1
                )
            )
            in_channels = adapt_channels

        channels_full = [in_channels] + list(channels)

        for i in range(n_layers):
            layers.append(
                Upsampler(
                    in_channels=channels_full[i],
                    out_channels=channels_full[i + 1],
                    up_factor=up_factors[i],
                    n_residual_blocks=n_res_blocks[i],
                    residual_kernel_size=res_kernel_sizes[i],
                    residual_dilation=res_dilations[i],
                    activation=activation,
                )
            )

        self.decoder = torch.nn.Sequential(*layers)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"\t[Decoder] Input: {x.shape}")

        for i, layer in enumerate(self.decoder):
            x = layer(x)
            print(f"\t[Decoder] Layer {i} ({layer.__class__.__name__}): {x.shape}")

        return x
