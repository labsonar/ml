"""
Module containing a Multi-Layer Perceptron (MLP) based models.
"""
import functools
import typing
import torch
import torch.nn

import lightning

class MLP(lightning.LightningModule):
    """ Multi-Layer Perceptron (MLP) implemented using PyTorch Lightning. """

    def __init__(
            self,
            input_shape: typing.Union[int, typing.Iterable[int]],
            hidden_channels: typing.Union[int, typing.Iterable[int]],
            n_targets: int = 2,
            norm_layer: typing.Optional[typing.Callable[..., torch.nn.Module]] = None,
            activation_layer: typing.Optional[typing.Callable[..., torch.nn.Module]] = None,
            activation_output_layer: typing.Optional[typing.Callable[..., torch.nn.Module]] = None,
            loss_fn: typing.Optional[typing.Callable[[], torch.nn.Module]] = None,
            bias: bool = True,
            dropout: float = 0.0,
            lr: float = 1e-3,
            weight_decay: float = 1e-4
        ):
        super().__init__()

        self.save_hyperparameters(ignore=["norm_layer",
                                          "activation_layer",
                                          "activation_output_layer",
                                          "loss_fn"])

        norm_layer = norm_layer or torch.nn.BatchNorm1d
        activation_layer = activation_layer or torch.nn.ReLU
        activation_output_layer = activation_output_layer or torch.nn.Sigmoid

        n_outputs = 1 if n_targets <= 2 else n_targets

        if loss_fn is None:
            if n_outputs == 1:
                loss_fn = torch.nn.BCEWithLogitsLoss
            else:
                loss_fn = torch.nn.CrossEntropyLoss

        if isinstance(input_shape, int):
            input_dim = input_shape
        else:
            input_dim = functools.reduce(lambda x, y: x * y, input_shape)

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

        layers = [torch.nn.Flatten(1)]
        in_dim = input_dim
        for hidden_dim in hidden_channels:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer())
            if dropout != 0:
                layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, n_outputs, bias=bias))

        if activation_output_layer is not None:
            layers.append(activation_output_layer())

        self.model = torch.nn.Sequential(*layers)
        self.loss_fn = loss_fn()
        self.lr = lr
        self.weight_decay = weight_decay
        self.is_binary = n_outputs == 1

    #pylint: disable=W0221
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the MLP. """
        out = self.model(inputs)

        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def _shared_step(self, batch: typing.Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Shared step for training and validation."""
        x, y = batch
        if self.is_binary:
            y = y.float()
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def training_step(self,
                batch: typing.Tuple[torch.Tensor, torch.Tensor],
                _: int = 0) -> torch.Tensor:
        """Executes a single training step."""
        loss = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self,
                batch: typing.Tuple[torch.Tensor, torch.Tensor],
                _: int = 0) -> torch.Tensor:
        """Executes a single validation step."""
        loss = self._shared_step(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self,
                batch: typing.Tuple[torch.Tensor, torch.Tensor],
                _: int = 0) -> torch.Tensor:
        """ Executes a single test step. """
        return self._shared_step(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """ Defines and returns the optimizer used during training. """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
