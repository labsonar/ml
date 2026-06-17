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
        embed_dim,
    ):
        super().__init__()

        self.film = lps_embedding.FiLM(embed_dim, in_channels)

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

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        x = self.film(x, embedding)
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
        embed_dim
    ):
        super().__init__()

        self.film = lps_embedding.FiLM(embed_dim, out_channels + skip_channels)

        self.upsample = lps_conv1d.UpsamplingBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2*stride,
            stride=stride,
            padding=stride//2,
            activation=activation,
        )

        layers = []
        for i in range(n_conv_layers):
            layers.append(
                lps_conv1d.Conv1DBlock(
                    in_channels= (out_channels + skip_channels) if i == 0 else out_channels,
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

    def forward(self, x: torch.Tensor, skip: torch.Tensor, embedding: torch.Tensor):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.film(x, embedding)
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
        embed_dim: int = 128,
    ):
        super().__init__()

        self.input_proj = torch.nn.Conv1d(in_channels * 2, base_channels, kernel_size=1)

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
                    embed_dim=embed_dim
                )
            )

            self.skip_channels.append(in_ch)
            in_ch = out_ch

        self.bottleneck = lps_stack1d.ResidualStack(
            channels=in_ch,
            n_blocks=2 * num_res_blocks,
            activation=activation,
        )
        self.temb_bottleneck = lps_embedding.FiLM(embed_dim, in_ch)

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
                    embed_dim=embed_dim
                )
            )

            in_ch = out_ch

        self.output_proj = torch.nn.Conv1d(base_channels, in_channels, kernel_size=1)

    def forward(self, cond: torch.Tensor, target: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:

        #print("cond:", cond.shape)
        #print("target:", target.shape)
        #print("t:", t.shape)
        #print("t_emb:", t_emb.shape)

        x = torch.cat([target, cond], dim=1)
        #print("x:", x.shape)
        x = self.input_proj(x)
        #print("input:", x.shape)

        skips = []

        #print("\n=== ENCODER ===")
        for i, block in enumerate(self.encoder):
            x, skip = block(x, embedding)
            #print(f"[ENC {i}] x: {x.shape} | skip: {skip.shape}")
            skips.append(skip)

        #print("\n=== BOTTLENECK ===")
        #print("before bottleneck:", x.shape)

        x = self.temb_bottleneck(x, embedding)
        x = self.bottleneck(x)

        #print("after bottleneck :", x.shape)

        #print("\n=== DECODER ===")
        for block in self.decoder:
            skip = skips.pop()

            #print(f"\n[DEC {i}] BEFORE")
            #print("x   :", x.shape)
            #print("skip:", skip.shape)

            x = block(x, skip, embedding)

            #print(f"[DEC {i}] AFTER")
            #print("x   :", x.shape)

        x = self.output_proj(x)

        return x
