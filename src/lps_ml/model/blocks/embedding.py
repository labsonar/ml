import math
import torch

class SinusoidalPositionalEncoding(torch.nn.Module):
    """ Sinusoidal Positional Encoding - Attention Is All You Need
    https://arxiv.org/pdf/1706.03762
    """

    def __init__(self, time_embed_dim: int):
        super().__init__()

        assert time_embed_dim % 2 == 0

        self.register_buffer(
            "inv_freq",
            self._get_inv_freq(time_embed_dim)
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(time_embed_dim, time_embed_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embed_dim * 4, time_embed_dim)
        )

    def _get_inv_freq(self, dim):
        half_dim = dim // 2
        return torch.exp(
            -math.log(10000.0) *
            torch.arange(half_dim).float() / half_dim
        )

    def forward(self, t):
        t = t.float()
        sinusoid_in = torch.outer(t, self.inv_freq)
        t_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        return self.mlp(t_emb) # (B, D)

class FiLM(torch.nn.Module):

    def __init__(self, time_embed_dim, channels):
        super().__init__()

        self.to_gamma_beta = torch.nn.Linear(time_embed_dim, channels * 2)

    def forward(self, x, t_emb):
        cond = self.to_gamma_beta(t_emb).unsqueeze(-1)
        gamma, beta = cond.chunk(2, dim=1)
        return (1 + gamma) * x + beta
