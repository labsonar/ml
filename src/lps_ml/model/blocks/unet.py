import typing
import torch

import lps_ml.model.blocks.embedding as lps_embedding
import lps_ml.model.blocks.conv1d as lps_conv1d
import lps_ml.model.blocks.stack1d as lps_stack1d


class EncoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_conv_layers,
        kernel_size,
        stride,
        activation,
        norm,
        time_embed_dim,
    ):
        super().__init__()

        self.film = lps_embedding.FiLM(time_embed_dim, in_channels)

        layers = []
        for _ in range(n_conv_layers):
            layers.append(
                lps_conv1d.Conv1DBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    activation=activation,
                    norm=norm,
                )
            )

        self.conv_block = torch.nn.Sequential(*layers)

        self.down = lps_conv1d.Conv1DBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                activation=activation,
                norm=norm,
            )

    def forward(self, x, t_emb):
        x = self.film(x, t_emb)
        skip = self.conv_block(x)
        y = self.down(skip)
        return y, skip

class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels,
        n_conv_layers,
        kernel_size,
        stride,
        activation,
        norm,
        num_res_blocks,
        time_embed_dim
    ):
        super().__init__()

        self.film = lps_embedding.FiLM(time_embed_dim, out_channels + skip_channels)

        self.upsample = lps_conv1d.UpsamplingBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2*stride,
            stride=stride,
            padding=stride//2,
            activation=activation,
        )

        out_channels = out_channels + skip_channels

        layers = []
        for _ in range(n_conv_layers):
            layers.append(
                lps_conv1d.Conv1DBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    activation=activation,
                    norm=norm,
                )
            )

        layers.append(
            lps_stack1d.ResidualStack(
                channels=out_channels,
                n_blocks=num_res_blocks,
                activation=activation,
            )
        )

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.film(x, t_emb)
        x = self.conv_block(x)
        return x

class UNet1D(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 128,
        channel_ratios: typing.List[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        kernel_size: int = 3,
        stride: int = 2,
        activation: typing.Callable = torch.nn.LeakyReLU,
        norm: typing.Optional[typing.Callable] = torch.nn.BatchNorm1d,
        n_internal_convs: int = 3,
        time_embed_dim: int = 128,
    ):
        super().__init__()

        self.input_proj = torch.nn.Conv1d(in_channels * 2, base_channels, kernel_size=1)
        self.time_embeder = lps_embedding.TimeEmbedding(time_embed_dim)

        self.encoder = torch.nn.ModuleList()

        self.skip_channels = []

        in_ch = base_channels

        for mult in channel_ratios:
            out_ch = base_channels * mult

            self.encoder.append(
                EncoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    n_conv_layers=n_internal_convs,
                    kernel_size=kernel_size,
                    stride=stride,
                    activation=activation,
                    norm=norm,
                    time_embed_dim=time_embed_dim
                )
            )

            self.skip_channels.append(out_ch)
            in_ch = out_ch

        self.bottleneck = lps_stack1d.ResidualStack(
            channels=in_ch,
            n_blocks=2 * num_res_blocks,
            activation=activation,
        )
        self.temb_bottleneck = lps_embedding.FiLM(in_ch, time_embed_dim)

        self.decoder = torch.nn.ModuleList()

        for mult, skip_ch in reversed(list(zip(channel_ratios, self.skip_channels))):
            out_ch = base_channels * mult

            self.decoder.append(
                DecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    skip_channels=skip_ch,
                    n_conv_layers=n_internal_convs,
                    kernel_size=kernel_size,
                    stride=stride,
                    activation=activation,
                    num_res_blocks=num_res_blocks,
                    norm=norm,
                    time_embed_dim=time_embed_dim
                )
            )

            in_ch = out_ch

        self.output_proj = torch.nn.Conv1d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        t_emb = self.time_embed(t)

        x = torch.cat([x, cond], dim=1)
        x = self.input_proj(x)

        skips = []

        for block in self.encoder:
            x, skip = block(x, t_emb)
            skips.append(skip)

        x = self.temb_bottleneck(x, t_emb)
        x = self.bottleneck(x)

        for block in self.decoder:
            skip = skips.pop()
            x = block(x, skip, t_emb)

        x = self.output_proj(x)

        return x
