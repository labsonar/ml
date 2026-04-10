import typing
import torch

from lps_utils.log import warning
class Conv1DBlock(torch.nn.Module):
    """
    1D convolutional downsampling block.

    Structure:
        Normalization -> Activation -> Dropout -> Conv1d (Pre-Activation Style)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        dilation: int = 1,
        padding: typing.Optional[int] = None,
        activation: typing.Union[typing.Callable, torch.nn.Module] = None,
        norm: typing.Optional[typing.Callable] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        activation_fn = activation or torch.nn.LeakyReLU
        norm_fn = norm or torch.nn.BatchNorm1d

        if padding is None:
            effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
            padding = (effective_kernel - 1) // 2

        layers = []

        if norm_fn is not None:
            layers.append(norm_fn(in_channels))

        if activation_fn is not None:
            layers.append(activation_fn() if isinstance(activation_fn, type) else activation_fn)

        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout))

        layers.append(
            torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
        )

        self.block = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UpsamplingBlock(torch.nn.Module):
    """
    1D upsampling block.

    Structure:
        Activation -> ConvTranspose1d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: typing.Optional[int] = None,
        output_padding: typing.Optional[int] = None,
        activation: typing.Union[typing.Callable, torch.nn.Module] = None,
    ):
        super().__init__()

        activation = activation or torch.nn.LeakyReLU

        if padding is None:
            padding = (kernel_size - stride) // 2

        if output_padding is None:
            output_padding = 0

        self.block = torch.nn.Sequential(
            activation(),
            torch.nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ResidualBlock(torch.nn.Module):
    """
    1D residual block.

    Structure:
        x + Conv1d(Activation(x))
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        padding: typing.Optional[int] = None,
        activation: typing.Union[typing.Callable, torch.nn.Module] = None,
    ):
        super().__init__()

        activation = activation or torch.nn.LeakyReLU

        if padding is None:
            effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
            padding = (effective_kernel - 1) // 2

            if effective_kernel % 2 == 0:
                warning(
                    "For ResidualBlock1D the effective_kernel must be odd to ensure "
                    "the output has the same length as the input. "
                    f"Setting padding to {padding} for kernel_size={kernel_size} "
                    f"and dilation={dilation}. Making the effective kernel "
                    f"{effective_kernel}."
                )

        self.block = torch.nn.Sequential(
            activation(),
            torch.nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)
