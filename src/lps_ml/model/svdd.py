"""
Module containing a Multi-Layer Perceptron (MLP) based models.
"""
import enum
import typing
import functools

import numpy as np
import sklearn.neighbors as sk_neighbors

import torch
import torch.nn
import torch.utils.data as torch_data
import lightning

import lps_ml.utils.general as ml_utils

class SVDDLoss(enum.Enum):
    """SVDD objective functions."""
    SOFT_BOUNDARY = enum.auto()
    ONE_CLASS = enum.auto()

class SVDDMLP(lightning.LightningModule):
    """Deep SVDD using an MLP embedding network."""

    def __init__(
        self,
        input_shape: typing.Union[int, typing.Iterable[int]],
        hidden_channels: typing.Union[int, typing.Iterable[int]],
        latent_dim: int = 32,
        loss: SVDDLoss = SVDDLoss.SOFT_BOUNDARY,
        nu: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        center_eps: float = 1e-4,
        warm_up_n_epochs: int = 10,
    ):
        super().__init__()

        if not 0.0 < nu <= 1.0:
            raise ValueError("nu must be in the interval (0, 1].")

        if isinstance(input_shape, int):
            input_dim = input_shape
        else:
            input_dim = functools.reduce(lambda x, y: x * y, input_shape)

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

        self.save_hyperparameters()

        self.loss = loss
        self.nu = nu
        self.lr = lr
        self.weight_decay = weight_decay
        self.center_eps = center_eps
        self.warm_up_n_epochs = warm_up_n_epochs
        self._epoch_distances = []

        layers : list[torch.nn.Module] = [torch.nn.Flatten(1)]

        in_dim = input_dim

        for hidden_dim in hidden_channels:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=False))
            layers.append(torch.nn.ReLU())
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, latent_dim, bias=False))

        self.embedder = torch.nn.Sequential(*layers)

        self.register_buffer("radius", torch.tensor(0.0))
        self.register_buffer("center", torch.zeros(latent_dim))
        self.register_buffer("center_initialized", torch.tensor(False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedder(x)

    def configure_optimizers(self):

        return torch.optim.Adam(
            self.embedder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    @staticmethod
    def _get_input(batch) -> torch.Tensor:
        """Extract input from a dataloader batch."""
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        return x

    @torch.no_grad()
    def initialize_center(self, dataloader) -> None:
        """
        Initialize the hypersphere center from the initial network.
        """

        embeddings = []

        with ml_utils.evaluating(self.embedder):

            for batch in dataloader:

                x = self._get_input(batch)
                x = x.to(self.device)

                z = self(x)
                embeddings.append(z.detach())

            if not embeddings:
                raise RuntimeError("Training dataloader produced no samples.")

            embeddings = torch.cat(embeddings, dim=0)
            center = embeddings.mean(dim=0)

            offseted_center = (torch.sign(center) + (center == 0).to(center.dtype)) * self.center_eps
            center = torch.where(center.abs() < self.center_eps, offseted_center, center)

            self.center.copy_(center)
            self.center_initialized.fill_(True)

    @torch.no_grad()
    def update_radius(self) -> None:
        """
        Update the hypersphere radius.
        """

        if self.loss != SVDDLoss.SOFT_BOUNDARY:
            return

        if not self._epoch_distances:
            return

        distances = torch.cat(self._epoch_distances, dim=0)
        distances = torch.sqrt(torch.clamp(distances, min=0.0))
        radius = torch.quantile(distances, 1.0 - self.nu)

        self.radius.copy_(radius)

    def squared_distance(self, z: torch.Tensor) -> torch.Tensor:
        """Squared distance between embeddings and SVDD center."""

        if not self.center_initialized.item():
            raise RuntimeError("SVDD center has not been initialized.")

        return torch.sum((z - self.center) ** 2, dim=-1)

    def compute_loss(self, squared_distances: torch.Tensor) -> \
            typing.Tuple[torch.Tensor, typing.Dict[str, torch.Tensor]]:
        """Compute the selected SVDD loss and its components, for logging."""

        if self.loss == SVDDLoss.ONE_CLASS:
            return squared_distances.mean(), {}

        if self.loss == SVDDLoss.SOFT_BOUNDARY:

            radius_squared = self.radius.square()
            penalty = torch.mean(torch.clamp(squared_distances - radius_squared, min=0.0)) / self.nu

            loss = radius_squared + penalty
            components = {
                "radius_term": radius_squared.detach(),
                "penalty_term": penalty.detach(),
            }

            return loss, components

        raise RuntimeError(f"Unsupported SVDD loss: {self.loss}")

    def on_train_epoch_end(self):

        if self.loss == SVDDLoss.SOFT_BOUNDARY:

            if self.current_epoch >= self.warm_up_n_epochs:
                self.update_radius()

            self.log(
                "radius",
                self.radius,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self._epoch_distances.clear()

    # def on_fit_start(self) -> None:
    #     """
    #     Initialize the SVDD center before the first optimization step.
    #     """

    #     if self.center_initialized.item():
    #         return

    #     if self.trainer.datamodule is not None:
    #         dataloader = self.trainer.datamodule.train_dataloader()

    #     elif self.trainer.train_dataloader is not None:
    #         dataloader = self.trainer.train_dataloader

    #     else:
    #         raise RuntimeError(
    #             "Could not obtain training dataloader to "
    #             "initialize the SVDD center."
    #         )

    #     self.initialize_center(dataloader)

    # def on_train_start(self) -> None:
    #     """
    #     Initialize the SVDD center before the first training epoch.
    #     """

    #     if self.center_initialized.item():
    #         return

    #     dataloader = self.trainer.train_dataloader

    #     if dataloader is None:
    #         raise RuntimeError(
    #             "Could not obtain training dataloader "
    #             "to initialize the SVDD center."
    #         )

    #     # Lightning may wrap the dataloader in a list.
    #     if isinstance(dataloader, (list, tuple)):
    #         if len(dataloader) != 1:
    #             raise RuntimeError(
    #                 "SVDDMLP expects exactly one training dataloader."
    #             )

    #         dataloader = dataloader[0]

    #     self.initialize_center(dataloader)

    def _shared_step(self, batch, stage: str) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Shared training and validation step."""

        x = self._get_input(batch)
        z = self(x)

        squared_distances = self.squared_distance(z)
        loss, components = self.compute_loss(squared_distances)

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
        )
        for name, value in components.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=False,
            )

        return loss, squared_distances

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Execute a training step."""
        loss, squared_distances = self._shared_step(batch, stage="train")

        if self.loss == SVDDLoss.SOFT_BOUNDARY:
            self._epoch_distances.append(squared_distances.detach())

        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """Execute a validation step."""
        return self._shared_step(batch, stage="val")[0]

    @torch.no_grad()
    def _extract_embeddings(self, dataloader: torch_data.DataLoader) -> np.ndarray:
        """Extract embeddings from a dataloader."""

        embeddings = []

        with ml_utils.evaluating(self.embedder):

            for batch in dataloader:

                x = self._get_input(batch)
                x = x.to(self.device)

                z = self(x)

                embeddings.append(z.detach().cpu())

        if not embeddings:
            raise RuntimeError("Dataloader produced no samples.")

        return torch.cat(embeddings, dim=0).numpy()


    @torch.no_grad()
    def calculate_alpha_beta_authenticity(
        self,
        real_dataloader: torch_data.DataLoader,
        synthetic_dataloader: torch_data.DataLoader,
        n_steps: int = 30,
    ) -> dict:
        """
        Evaluate alpha-precision, beta-recall and authenticity
        in the learned SVDD embedding space.
        Returns
        -------
        dict
            Dictionary containing:
                - alpha_precision
                - beta_recall
                - authenticity
        """

        if n_steps < 2:
            raise ValueError("n_steps must be at least 2.")

        real_data = self._extract_embeddings(real_dataloader).astype(np.float32)
        synthetic_data = self._extract_embeddings(synthetic_dataloader).astype(np.float32)

        center = self.center.detach().cpu().numpy().astype(np.float32)

        alphas = np.linspace(0.0, 1.0, n_steps)

        # Distance of real samples to SVDD center
        real_to_center = np.linalg.norm(real_data - center, axis=1)
        radii = np.quantile(real_to_center, alphas)

        # Distance of synthetic samples to SVDD center
        synthetic_to_center = np.linalg.norm(synthetic_data - center, axis=1)
        synthetic_center = np.mean(synthetic_data, axis=0)

        # Real -> nearest real sample
        real_nn = sk_neighbors.NearestNeighbors(n_neighbors=2, n_jobs=-1, metric="euclidean")
        real_nn.fit(real_data)
        real_distances, _ = real_nn.kneighbors(real_data)
        real_to_real = real_distances[:, 1]

        synthetic_nn = sk_neighbors.NearestNeighbors(n_neighbors=1, n_jobs=-1, metric="euclidean")
        synthetic_nn.fit(synthetic_data)

        real_to_synthetic_distances, indices = synthetic_nn.kneighbors(real_data)
        real_to_synthetic = real_to_synthetic_distances[:, 0]
        indices = indices[:, 0]

        closest_synthetic = synthetic_data[indices]

        closest_synthetic_distance_to_center = np.linalg.norm(
            closest_synthetic - synthetic_center,
            axis=1,
        )

        synthetic_radii = np.quantile(closest_synthetic_distance_to_center, alphas)

        alpha_precision_curve = []
        beta_recall_curve = []

        for radius, synthetic_radius in zip(radii, synthetic_radii):

            alpha_precision = np.mean(synthetic_to_center <= radius)

            beta_recall = np.mean(
                (real_to_synthetic <= real_to_real)
                &
                (closest_synthetic_distance_to_center <= synthetic_radius)
            )

            alpha_precision_curve.append(alpha_precision)
            beta_recall_curve.append(beta_recall)

        alpha_precision_curve = np.asarray(alpha_precision_curve, dtype=np.float64)
        beta_recall_curve = np.asarray(beta_recall_curve, dtype=np.float64)

        synthetic_to_real_nn = sk_neighbors.NearestNeighbors(
            n_neighbors=1,
            n_jobs=-1,
            metric="euclidean",
        )

        synthetic_to_real_nn.fit(real_data)

        synthetic_to_real_distances, real_indices = synthetic_to_real_nn.kneighbors(synthetic_data)
        synthetic_to_real = synthetic_to_real_distances[:, 0]

        real_indices = real_indices[:, 0]
        nearest_real_to_real = real_to_real[real_indices]

        authenticity = np.mean(synthetic_to_real > nearest_real_to_real)

        delta_alpha_precision = \
            1.0 - 2.0 * np.sum(np.abs(alphas - alpha_precision_curve)) * (alphas[1] - alphas[0])

        delta_beta_recall = \
            1.0 - 2.0 * np.sum(np.abs(alphas - beta_recall_curve)) * (alphas[1] - alphas[0])

        # 1) Checar variância/colapso dos embeddings
        print("std distância real->centro:", real_to_center.std(), "média:", real_to_center.mean())
        print("std distância synth->centro:", synthetic_to_center.std(), "média:", synthetic_to_center.mean())
        # se std << média, hiperesfera pode estar colapsando

        # 2) Checar duplicatas/empates
        print("valores únicos em real_synth_closest_d:", len(np.unique(closest_synthetic_distance_to_center)),
            "de", len(closest_synthetic_distance_to_center))
        print("valores únicos em synthetic_data:", len(np.unique(synthetic_data, axis=0)), "de", len(synthetic_data))

        # 3) Checar monotonicidade da curva (deveria ser não-decrescente)
        print("beta_recall_curve monotônica?", np.all(np.diff(beta_recall_curve) >= -1e-9))

        # 4) Checar se curve(alpha) > alpha em algum ponto (viola a invariante)
        print("pontos onde curve > alpha:", np.where(beta_recall_curve > alphas + 1e-9)[0])

        return {
            "alpha_precision": float(delta_alpha_precision),
            "beta_recall": float(delta_beta_recall),
            "authenticity": float(authenticity),

            "alphas": alphas,
            "alpha_precision_curve": alpha_precision_curve,
            "beta_recall_curve": beta_recall_curve,
        }
