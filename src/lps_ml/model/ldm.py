import enum
import typing
import math

import torch
import lightning

import lps_ml.model.blocks.unet as ml_unet

class LDMLoss(enum.Enum):
    """ Loss functions for training the Latent Diffusion Model. """
    MSE = enum.auto()
    L2 = enum.auto()
    HUBER = enum.auto()
    CHARBONNIER = enum.auto()

class LatentDiffusionModel(lightning.LightningModule):

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

        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        lr: float = 1e-4,
        loss: LDMLoss = LDMLoss.MSE
    ):
        super().__init__()
        self.save_hyperparameters()

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
            time_embed_dim=time_embed_dim,
        )

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

    def forward(self, cond: torch.Tensor, target: torch.Tensor, t: torch.Tensor):
        """
        Predict noise ε
        """
        return self.unet(cond=cond, target=target, t=t)

    def _shared_step(self, batch, stage: str):

        data, _ = batch
        x_cond = data[0]
        x_target = data[1]

        batch_size = x_target.shape[0]
        device = x_target.device

        t = torch.randint(0, self.timesteps, (batch_size,), device=device)

        noise = torch.randn_like(x_target)
        x_noisy = self.q_sample(x_target, t, noise)

        noise_pred = self.forward(cond=x_cond,
                                  target=x_noisy,
                                  t=t)

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
    def sample(self, cond):

        x = torch.randn(cond.shape, device=cond.device)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)

            noise_pred = self.unet(cond=cond,
                                   target=x,
                                   t=t_tensor)

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

LDM = LatentDiffusionModel