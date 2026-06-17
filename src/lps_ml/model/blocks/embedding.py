import abc
import typing
import math
import torch

class Embedder(torch.nn.Module):
    """ Basic Embedder abstraction """

    def __init__(self, embed_dim: int, up_factor: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.up_factor = up_factor

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * up_factor),
            torch.nn.SiLU(),
            torch.nn.Linear(embed_dim * up_factor, embed_dim)
        )

    @abc.abstractmethod
    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """ Embeding abstract method
        Args:
            x (torch.Tensor): input tensor in shape (Batch,)

        Returns:
            torch.Tensor: output tensor in shape (Batch, Embed_dim)
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Embed an x input """
        embedded = self._embed(x)
        return self.mlp(embedded) #(Batch, Embed_dim)

class ContinuousSinusoidalEmbedder(Embedder):
    """ ContinuousSinusoidalEmbedder """

    def __init__(self,
                 embed_dim: int,
                 up_factor: int = 4,
                 min_value: float = 0.0,
                 max_value: float = 1.0,
                 scale: float = 1000.0,
                 ):
        assert embed_dim % 2 == 0

        super().__init__(embed_dim, up_factor)

        self.min = min_value
        self.max = max_value
        self.scale = scale

        self.register_buffer(
            "inv_freq",
            self._get_inv_freq(embed_dim)
        )

    def _get_inv_freq(self, dim):
        half_dim = dim // 2
        return torch.exp(
            -math.log(10000.0) *
            torch.arange(half_dim).float() / half_dim
        )

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        scaled_value = self.scale * torch.clamp(x, min=self.min, max=self.max)
        sinusoid_in = torch.outer(scaled_value, self.inv_freq)
        t_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        return t_emb # (B, D)

class CategoricalEncoder(Embedder):
    """ CategoricalEncoder """

    def __init__(self,
                 n_classes: int,
                 embed_dim: int,
                 up_factor: int = 4):

        super().__init__(embed_dim, up_factor)

        self.embedding = torch.nn.Embedding(n_classes, embed_dim)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x.long()) # (B, D)

class FusionEmbedder(torch.nn.Module):
    """ FusionEmbedder """

    def __init__(self,
                 embedders: typing.Dict[str, Embedder],
                 embed_dim: int,
                 up_factor: int = 4):
        super().__init__()
        self.embedders = torch.nn.ModuleDict(embedders)
        self.embed_dim = embed_dim

        input_dim = sum(embedder.embed_dim for embedder in embedders.values())

        self.mlp_fusion = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embed_dim * up_factor),
            torch.nn.SiLU(),
            torch.nn.Linear(embed_dim * up_factor, embed_dim)
        )

    def forward(self, conditions: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
        """ Embed an x input """

        embeddings = []

        for name, embedder in self.embedders.items():

            if name not in conditions:
                raise ValueError(f"Missing condition '{name}'")

            embeddings.append(embedder(conditions[name]))

        if len(embeddings) == 1:
            return embeddings[0]
        return self.mlp_fusion(torch.cat(embeddings, dim=-1))

class FiLM(torch.nn.Module):
    """ FiLM layer for adapte embedder """

    def __init__(self, embed_dim: int, channels: int):
        super().__init__()

        self.to_gamma_beta = torch.nn.Linear(embed_dim, channels * 2)

    def forward(self, data: torch.Tensor, embedded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            data (torch.Tensor): data in shape (Batch, channels, samples)
            embedded (torch.Tensor): (Batch, embed_dim)

        Returns:
            torch.Tensor: transformed data in shape (Batch, channels, samples)
        """
        cond = self.to_gamma_beta(embedded).unsqueeze(-1)
        gamma, beta = cond.chunk(2, dim=1)
        return (1 + gamma) * data + beta
