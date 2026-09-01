import enum
import typing
import argparse

import torch
import lightning

import lps_ml.model.blocks.embedding as ml_emb
import lps_ml.model.blocks.unet as ml_unet
import lps_ml.core.datamodule as ml_core

class LDMLoss(enum.Enum):
    """ Loss functions for training the Latent Diffusion Model. """
    MSE = enum.auto()
    L2 = enum.auto()
    HUBER = enum.auto()
    CHARBONNIER = enum.auto()

class Condition(enum.Enum):
    TIME = enum.auto()
    DISTANCE = enum.auto()
    INPUT_CHANNEL = enum.auto()
    OUTPUT_CHANNEL = enum.auto()

class ChannelMode(enum.Enum):
    FIXED = enum.auto()
    VARIABLE = enum.auto()

class LatentDiffusionModel(lightning.LightningModule):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        n_channels: int,
        channel_mode: ChannelMode = ChannelMode.FIXED,
        fixed_input_channel: int = 0,
        fixed_output_channel: int = 1,
        embeed_distance: bool = False,

        base_channels: int = 128,
        channel_ratios: typing.List[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        kernel_size: int = 3,
        stride: int = 2,
        activation: typing.Callable = torch.nn.LeakyReLU,
        norm: typing.Optional[typing.Callable] = torch.nn.BatchNorm1d,
        n_internal_convs: int = 3,

        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        lr: float = 1e-4,
        loss: LDMLoss = LDMLoss.MSE
    ):
        super().__init__()
        self.save_hyperparameters()

        embedders : typing.Dict[str, ml_emb.Embedder] = {
            Condition.TIME.name: ml_emb.ContinuousSinusoidalEmbedder(
                    embed_dim,
                    max_value=timesteps
                )
            }

        if channel_mode == ChannelMode.VARIABLE:
            embedders[Condition.INPUT_CHANNEL.name] = ml_emb.CategoricalEncoder(
                    n_classes=n_channels,
                    embed_dim=embed_dim,
                )
            embedders[Condition.OUTPUT_CHANNEL.name] = ml_emb.CategoricalEncoder(
                    n_classes=n_channels,
                    embed_dim=embed_dim,
                )

        if embeed_distance:
            embedders[Condition.DISTANCE.name] = ml_emb.ContinuousSinusoidalEmbedder(
                    embed_dim,
                    min_value=50,
                    max_value=150
                )

        conditioning_embedder = ml_emb.FusionEmbedder(embedders, embed_dim)

        self.unet = ml_unet.UNet1D(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_ratios=channel_ratios,
            num_res_blocks=num_res_blocks,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            norm=norm,
            n_internal_convs=n_internal_convs,
            embed_dim=conditioning_embedder.embed_dim,
        )

        self.n_channels = n_channels
        self.channel_mode = channel_mode
        self.fixed_input_channel = fixed_input_channel
        self.fixed_output_channel = fixed_output_channel

        self.conditioning_embedder = conditioning_embedder
        self.timesteps = timesteps
        self.lr = lr
        self.loss = loss

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_hat", alpha_hat)

    def q_sample(self, x0, t, noise):
        """
        x_t = sqrt(alpha_hat) * x0 + sqrt(1 - alpha_hat) * noise
        """

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]

        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise

    def forward(self,
                cond: torch.Tensor,
                target: torch.Tensor,
                conditions: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
        embedding = self.conditioning_embedder(conditions)
        return self.unet(cond=cond, target=target, embedding=embedding)

    def _shared_step(self, batch, stage: str):

        data, target, _ = batch
        data = torch.stack(data, dim=1)

        batch_size = data.shape[0]

        if self.channel_mode == ChannelMode.FIXED:
            in_ch = torch.full((batch_size,), self.fixed_input_channel, device=data.device)
            out_ch = torch.full((batch_size,), self.fixed_output_channel, device=data.device)

        else:
            in_ch = torch.randint(0, self.n_channels, (batch_size,), device=data.device)
            out_ch = torch.randint(0, self.n_channels - 1, (batch_size,), device=data.device)
            out_ch += (out_ch >= in_ch)

        batch_idx = torch.arange(batch_size, device=in_ch.device)

        x_cond = data[batch_idx, in_ch]
        x_target = data[batch_idx, out_ch]

        t = torch.randint(0, self.timesteps, (batch_size,), device=data.device)

        conditions = {
            Condition.TIME.name: t.float(),
            Condition.DISTANCE.name: target.float(),
            Condition.INPUT_CHANNEL.name: in_ch,
            Condition.OUTPUT_CHANNEL.name: out_ch,
        }

        noise = torch.randn_like(x_target)
        x_noisy = self.q_sample(x_target, t, noise)

        noise_pred = self.forward(cond=x_cond,
                                  target=x_noisy,
                                  conditions=conditions)

        if self.loss == LDMLoss.MSE:
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
        elif self.loss == LDMLoss.L2:
            loss = torch.mean(torch.sqrt(
                torch.sum((noise_pred - noise) ** 2,
                          dim=tuple(range(1, noise_pred.ndim)))
            ))
        elif self.loss == LDMLoss.HUBER:
            loss = torch.nn.functional.smooth_l1_loss(noise_pred, noise)
        elif self.loss == LDMLoss.CHARBONNIER:
            loss = torch.mean(torch.sqrt((noise_pred - noise) ** 2 + 1e-3 ** 2))
        else:
            raise ValueError(f"Unsupported loss type: {self.loss}")

        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, distance: torch.Tensor | None = None, input_ch: int = 0, output_ch: int = 1):

        x = torch.randn(cond.shape, device=cond.device)

        conditions = {
            Condition.INPUT_CHANNEL.name: torch.full(
                (x.shape[0],),
                input_ch,
                device=x.device,
                dtype=torch.long,
            ),
            Condition.OUTPUT_CHANNEL.name: torch.full(
                (x.shape[0],),
                output_ch,
                device=x.device,
                dtype=torch.long,
            ),
        }

        if distance is not None:
            conditions[Condition.DISTANCE.name] = distance.float()


        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.float32)

            conditions[Condition.TIME.name] = t_tensor

            noise_pred = self(cond=cond,
                              target=x,
                              conditions=conditions)

            alpha = self.alphas[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (
                1 / torch.sqrt(alpha) *
                (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * noise_pred)
                + torch.sqrt(beta) * noise
            )

        return x

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
        """Add Latent Diffusion Model arguments to an argparse parser."""

        group = parser.add_argument_group("LDM")

        # Diffusion
        group.add_argument( "--ldm-steps", type=int, default=300,
            help="Number of diffusion timesteps.")

        group.add_argument( "--beta-start", type=float, default=1e-4,
            help="Initial value of the diffusion beta schedule.")

        group.add_argument( "--beta-end", type=float, default=0.02,
            help="Final value of the diffusion beta schedule.")

        group.add_argument( "--ldm-loss", type=str, choices=[loss.name.lower() for loss in LDMLoss],
            default=LDMLoss.MSE.name.lower(), help="Loss function used to train the LDM.")

        # Conditioning
        group.add_argument( "--embed-dim", type=int, default=128,
            help="Dimension of the conditioning embedding.")

        group.add_argument( "--channel-mode", type=str.lower,
            choices=[mode.name.lower() for mode in ChannelMode],
            default=ChannelMode.FIXED.name.lower(),
            help="Channel conditioning mode."
        )

        group.add_argument( "--fixed-input-channel", type=int, default=0,
            help="Input channel used when channel mode is FIXED.")

        group.add_argument( "--fixed-output-channel", type=int, default=1,
            help="Output channel used when channel mode is FIXED.")

        group.add_argument("--embed-distance", action="store_true",
            help="Enable distance conditioning.")

        # U-Net
        group.add_argument( "--base-channels", type=int, default=128,
            help="Base number of channels in the U-Net.")

        group.add_argument( "--channel-ratios", type=int, nargs="+", default=[1, 2, 4],
            help="Channel multipliers for each U-Net level.")

        group.add_argument( "--num-res-blocks", type=int, default=2,
            help="Number of residual blocks per U-Net level.")

        group.add_argument( "--kernel-size", type=int, default=3,
            help="Kernel size used by the U-Net.")

        group.add_argument( "--stride", type=int, default=2,
            help="Stride used by the U-Net down/up-sampling blocks.")

        group.add_argument( "--n-internal-convs", type=int, default=3,
            help="Number of internal convolutions in each U-Net block.")

        group.add_argument( "--ldm-activation", type=str,
            choices=["relu", "leaky_relu", "gelu", "tanh"], default="leaky_relu",
            help="Activation function used by the U-Net."
        )

        group.add_argument( "--ldm-norm", type=str, choices=["batch_norm", "none"],
            default="batch_norm", help="Normalization layer used by the U-Net.")

        # Optimization
        group.add_argument( "--ldm-lr", type=float, default=1e-4, help="Learning rate.")

        return group

    @staticmethod
    def from_args(
        args: argparse.Namespace,
        dm: ml_core.BaseDataModule,
    ) -> "LatentDiffusionModel":
        """Create a LatentDiffusionModel from command-line arguments."""

        activation_layers = {
            "relu": torch.nn.ReLU,
            "leaky_relu": torch.nn.LeakyReLU,
            "gelu": torch.nn.GELU,
            "tanh": torch.nn.Tanh,
        }

        norm_layers = {
            "batch_norm": torch.nn.BatchNorm1d,
            "none": None,
        }

        print("dm.get_sample_shape(): ", dm.get_sample_shape())

        return LatentDiffusionModel(
            in_channels=dm.get_sample_shape()[0],
            embed_dim=args.embed_dim,
            n_channels=dm.get_n_channels(),

            channel_mode=ChannelMode[args.channel_mode.upper()],
            fixed_input_channel=args.fixed_input_channel,
            fixed_output_channel=args.fixed_output_channel,
            embeed_distance=args.embed_distance,

            base_channels=args.base_channels,
            channel_ratios=args.channel_ratios,
            num_res_blocks=args.num_res_blocks,
            kernel_size=args.kernel_size,
            stride=args.stride,
            activation=activation_layers[args.ldm_activation],
            norm=norm_layers[args.ldm_norm],
            n_internal_convs=args.n_internal_convs,

            timesteps=args.ldm_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,

            lr=args.ldm_lr,
            loss=LDMLoss[args.ldm_loss.upper()],
        )

    def get_pairs(self) -> typing.List[typing.Tuple[int, int]]:
        """ Generate all pair combinations of input and output channels trained in model. """

        if self.channel_mode == ChannelMode.FIXED:
            channel_pairs = [
                (
                    self.fixed_input_channel,
                    self.fixed_output_channel
                )
            ]

        else:
            channel_pairs = [
                (i, j)
                for i in range(self.n_channels)
                for j in range(self.n_channels)
                if i != j
            ]

        return channel_pairs

LDM = LatentDiffusionModel