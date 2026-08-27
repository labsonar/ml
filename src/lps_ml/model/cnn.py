"""
Module containing a Convolutional Neural Network (CNN) based models.
"""
import typing
import argparse
import torch

import lps_ml.core.datamodule as ml_core
import lps_ml.model.mlp as lps_mlp

class CNN1D(lps_mlp.MLP):
    """CNN with MLP head, compatible with binary or multiclass classification."""

    def __init__(
        self,
        input_shape: typing.Iterable[int],

        conv_n_neurons: typing.List[int],
        conv_activation: typing.Union[torch.nn.Module, typing.Callable] = None,
        conv_pooling: typing.Optional[typing.Callable] = None,
        conv_pooling_size: typing.Union[int, typing.List[int]] = 2,
        conv_dilation: typing.Union[int, typing.List[int]] = 1,
        conv_dropout: float = 0.5,
        batch_norm: typing.Optional[typing.Callable] = None,
        kernel_size: int = 5,
        padding: int = None,

        classification_n_neurons: typing.Union[int, typing.Iterable[int]] = 128,
        n_targets: int = 2,
        classification_dropout: float = 0,
        classification_norm: typing.Optional[typing.Callable] = None,
        classification_hidden_activation: typing.Optional[typing.Callable] = None,
        classification_output_activation: typing.Optional[typing.Callable] = None,
        lr: float = 1e-3,
        loss_fn: typing.Optional[typing.Callable[[], torch.nn.Module]] = None,
        weight_decay: float = 1e-4
    ):

        conv_activation = conv_activation or torch.nn.ReLU
        conv_pooling = conv_pooling or torch.nn.MaxPool1d
        batch_norm = batch_norm or torch.nn.BatchNorm1d
        classification_norm = classification_norm or torch.nn.BatchNorm1d
        classification_hidden_activation = classification_hidden_activation or conv_activation

        input_shape = list(input_shape)

        if len(input_shape) == 1:
            input_shape = [1] + input_shape

        elif len(input_shape) == 2:
            if input_shape[0] != 1:
                raise ValueError(f"CNN1D expects input_shape as [X] or [1, X]. Got {input_shape}.")

        else:
            raise ValueError(f"CNN1D expects input_shape as [X] or [1, X]. Got {input_shape}.")

        if isinstance(conv_dilation, int):
            conv_dilation = [conv_dilation] * len(conv_n_neurons)
        elif len(conv_dilation) == 1:
            conv_dilation = conv_dilation * len(conv_n_neurons)
        elif len(conv_dilation) != len(conv_n_neurons):
            raise ValueError(
                "conv_dilation must contain either one value "
                "or one value per convolutional layer."
            )

        if isinstance(conv_pooling_size, int):
            conv_pooling_size = [conv_pooling_size] * len(conv_n_neurons)
        elif len(conv_pooling_size) == 1:
            conv_pooling_size = conv_pooling_size * len(conv_n_neurons)
        elif len(conv_pooling_size) != len(conv_n_neurons):
            raise ValueError(
                "conv_pooling_size must contain either one value "
                "or one value per convolutional layer."
            )

        if padding is None:
            padding = (kernel_size - 1) // 2

        conv_layers = []
        conv_channels = [input_shape[0]] + conv_n_neurons

        for i in range(1, len(conv_channels)):

            dilation = conv_dilation[i - 1]

            # Effective kernel size for dilated convolution
            effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
            current_padding = (effective_kernel - 1) // 2

            conv_layers.append(
                torch.nn.Conv1d(
                    in_channels=conv_channels[i - 1],
                    out_channels=conv_channels[i],
                    kernel_size=kernel_size,
                    padding=current_padding,
                    dilation=dilation,
                )
            )

            if batch_norm is not None:
                conv_layers.append(batch_norm(conv_channels[i]))

            if conv_dropout != 0:
                conv_layers.append(torch.nn.Dropout1d(p=conv_dropout))

            conv_layers.append(conv_activation())

            if conv_pooling is not None:
                conv_layers.append(conv_pooling(kernel_size=conv_pooling_size[i - 1]))

        conv_layers = torch.nn.Sequential(*conv_layers)
        test_tensor = torch.rand([1] + input_shape, dtype=torch.float32)
        features = conv_layers(test_tensor)

        super().__init__(
            input_shape=features.shape,
            hidden_channels=classification_n_neurons,
            norm_layer=classification_norm,
            n_targets=n_targets,
            activation_layer=classification_hidden_activation,
            activation_output_layer=classification_output_activation,
            dropout=classification_dropout,
            loss_fn=loss_fn,
            weight_decay=weight_decay,
            lr=lr
        )

        self.save_hyperparameters(ignore=[
            "conv_activation",
            "conv_pooling",
            "batch_norm",
            "classification_hidden_activation",
            "classification_output_activation",
            "loss_fn",
        ])

        self.conv_layers = conv_layers

    def to_feature_space(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through the convolutional part of the network."""
        return self.conv_layers(x)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN."""

        # [B, X] -> [B, 1, X]
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(1)

        elif inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError(
                "CNN1D expects input with shape [B, X] or [B, 1, X]. "
                f"Got {tuple(inputs.shape)}."
            )

        features = self.to_feature_space(inputs)
        out = super().forward(features)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
        """Add CNN1D arguments to an argparse parser."""

        group = parser.add_argument_group("CNN1D")

        group.add_argument( "--cnn1d-conv-n-neurons", type=int, nargs="+", default=[4, 8, 16, 32],
            help="Number of filters in each convolutional layer."
        )

        group.add_argument( "--cnn1d-conv-activation", type=str, default="relu",
            choices=["relu", "leaky_relu", "gelu", "tanh"],
        )

        group.add_argument( "--cnn1d-conv-pooling", type=str, default="max_pool",
            choices=["max_pool", "avg_pool", "none"],
        )

        group.add_argument( "--cnn1d-conv-pooling-size", type=int, nargs="+", default=[2])
        group.add_argument( "--cnn1d-conv-dilation", type=int, nargs="+", default=[1])
        group.add_argument( "--cnn1d-conv-dropout", type=float, default=0.5)

        group.add_argument( "--cnn1d-batch-norm", type=str, default="batch_norm",
            choices=["batch_norm", "none"],
        )

        group.add_argument( "--cnn1d-kernel-size", type=int, default=5)
        group.add_argument( "--cnn1d-padding", type=int, default=None)
        group.add_argument( "--cnn1d-classification-n-neurons", type=int, nargs="+", default=[64])
        group.add_argument( "--cnn1d-classification-dropout", type=float, default=0.0)

        group.add_argument( "--cnn1d-classification-norm", type=str, default="batch_norm",
            choices=["batch_norm", "none"],
        )

        group.add_argument( "--cnn1d-classification-hidden-activation", type=str, default=None,
            choices=["relu", "leaky_relu", "gelu", "tanh"],
        )

        group.add_argument( "--cnn1d-classification-output-activation", type=str, default="sigmoid",
            choices=["sigmoid", "softmax", "none"],
        )

        group.add_argument( "--cnn1d-lr", type=float, default=1e-3)
        group.add_argument( "--cnn1d-weight-decay", type=float, default=1e-4)

        return group

    @staticmethod
    def from_args(args: argparse.Namespace, dm: ml_core.BaseDataModule) -> "CNN1D":
        """Create a CNN1D from command-line arguments."""

        activations = {
            "relu": torch.nn.ReLU,
            "leaky_relu": torch.nn.LeakyReLU,
            "gelu": torch.nn.GELU,
            "tanh": torch.nn.Tanh,
        }

        pooling = {
            "max_pool": torch.nn.MaxPool1d,
            "avg_pool": torch.nn.AvgPool1d,
            "none": None,
        }

        norm = {
            "batch_norm": torch.nn.BatchNorm1d,
            "none": None,
        }

        output_activation = {
            "sigmoid": torch.nn.Sigmoid,
            "softmax": torch.nn.Softmax,
            "none": None,
        }

        hidden_activation = (
            activations[args.cnn1d_classification_hidden_activation]
            if args.cnn1d_classification_hidden_activation is not None
            else None
        )

        return CNN1D(
            input_shape=dm.get_sample_shape(),
            n_targets=dm.get_n_targets(),
            conv_n_neurons=args.cnn1d_conv_n_neurons,
            conv_activation=activations[args.cnn1d_conv_activation],
            conv_pooling=pooling[args.cnn1d_conv_pooling],
            conv_pooling_size=args.cnn1d_conv_pooling_size,
            conv_dilation=args.cnn1d_conv_dilation,
            conv_dropout=args.cnn1d_conv_dropout,
            batch_norm=norm[args.cnn1d_batch_norm],
            kernel_size=args.cnn1d_kernel_size,
            padding=args.cnn1d_padding,
            classification_n_neurons=args.cnn1d_classification_n_neurons,
            classification_dropout=args.cnn1d_classification_dropout,
            classification_norm=norm[args.cnn1d_classification_norm],
            classification_hidden_activation=hidden_activation,
            classification_output_activation=output_activation[
                args.cnn1d_classification_output_activation
            ],
            lr=args.cnn1d_lr,
            weight_decay=args.cnn1d_weight_decay,
        )

class CNN2D(lps_mlp.MLP):
    """ CNN with MLP head, compatible with binary or multiclass classification. """

    def __init__(
        self,
        input_shape: typing.Iterable[int],

        conv_n_neurons: typing.List[int],
        conv_activation: typing.Union[torch.nn.Module, typing.Callable] = None,
        conv_pooling: typing.Optional[typing.Callable] = None,
        conv_pooling_size: typing.List[int] = None,
        conv_dilation: typing.Union[int, typing.List[int]] = 1,
        conv_dropout: float = 0.5,
        batch_norm: typing.Optional[typing.Callable] = None,
        kernel_size: int = 5,
        padding: int = None,

        classification_n_neurons: typing.Union[int, typing.Iterable[int]] = 128,
        n_targets: int = 2,
        classification_dropout: float = 0,
        classification_norm: typing.Optional[typing.Callable] = None,
        classification_hidden_activation: typing.Optional[typing.Callable] = None,
        classification_output_activation: typing.Optional[typing.Callable] = None,
        loss_fn: typing.Optional[typing.Callable[[], torch.nn.Module]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        conv_activation = conv_activation or torch.nn.ReLU
        conv_pooling = conv_pooling or torch.nn.MaxPool2d
        conv_pooling_size = conv_pooling_size or [2, 2]
        batch_norm = batch_norm or torch.nn.BatchNorm2d
        classification_norm = classification_norm or torch.nn.BatchNorm1d

        classification_hidden_activation = classification_hidden_activation or conv_activation
        padding = padding or int((kernel_size - 1) / 2)

        if len(input_shape) == 2:
            input_shape = [1] + input_shape
        elif len(input_shape) != 3:
            raise ValueError(
                "CNN expects input in the format [C, H, W] or [H, W] "
                f"(current {input_shape})"
            )

        if isinstance(conv_dilation, int):
            conv_dilation = [conv_dilation] * len(conv_n_neurons)
        elif len(conv_dilation) == 1:
            conv_dilation = conv_dilation * len(conv_n_neurons)
        elif len(conv_dilation) != len(conv_n_neurons):
            raise ValueError(
                "conv_dilation must contain either one value "
                "or one value per convolutional layer."
            )

        conv_layers = []
        conv_channels = [input_shape[0]] + conv_n_neurons
        for i in range(1, len(conv_channels)):

            dilation = conv_dilation[i-1]
            effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
            current_padding = int((effective_kernel - 1) / 2)

            conv_layers.append(torch.nn.Conv2d(
                    conv_channels[i-1],
                    conv_channels[i],
                    kernel_size=kernel_size,
                    padding=current_padding,
                    dilation=dilation
                ))

            if batch_norm is not None:
                conv_layers.append(batch_norm(conv_channels[i]))
            if conv_dropout != 0 and i != 1:
                conv_layers.append(torch.nn.Dropout2d(p=conv_dropout))
            conv_layers.append(conv_activation())
            if conv_pooling is not None:
                conv_layers.append(conv_pooling(*conv_pooling_size))

        conv_layers = torch.nn.Sequential(*conv_layers)

        test_tensor = torch.rand([1] + list(input_shape), dtype=torch.float32)
        features = conv_layers(test_tensor)

        super().__init__(
            input_shape=features.shape,
            hidden_channels=classification_n_neurons,
            norm_layer=classification_norm,
            n_targets=n_targets,
            activation_layer=classification_hidden_activation,
            activation_output_layer=classification_output_activation,
            dropout=classification_dropout,
            loss_fn=loss_fn,
            weight_decay=weight_decay,
            lr=lr
        )

        self.save_hyperparameters(ignore=[
            "conv_activation", "conv_pooling", "batch_norm",
            "classification_hidden_activation", "classification_output_activation", "loss_fn"
        ])

        self.conv_layers = torch.nn.Sequential(*conv_layers)

    def to_feature_space(self, x: torch.Tensor) -> torch.Tensor:
        """ Pass the input through the convolutional part of the network."""
        return self.conv_layers(x)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the CNN. """

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)

        elif inputs.ndim != 4:
            raise ValueError(
                "CNN2D expects input with shape "
                "[B, H, W] or [B, C, H, W], "
                f"got {tuple(inputs.shape)}"
            )

        features = self.to_feature_space(inputs)
        out = super().forward(features)
        return out

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
        """Add CNN2D arguments to an argparse parser."""

        group = parser.add_argument_group("CNN2D")

        group.add_argument( "--cnn2d-conv-n-neurons", type=int, nargs="+", default=[16, 32, 64])

        group.add_argument( "--cnn2d-conv-activation", type=str, default="relu",
            choices=["relu", "leaky_relu", "gelu", "tanh"],
        )

        group.add_argument( "--cnn2d-conv-pooling", type=str, default="max_pool",
            choices=["max_pool", "avg_pool", "none"],
        )

        group.add_argument( "--cnn2d-conv-pooling-size", type=int, nargs="+", default=[4, 2])
        group.add_argument( "--cnn2d-conv-dilation", type=int, nargs="+", default=[1])
        group.add_argument( "--cnn2d-conv-dropout", type=float, default=0.5)

        group.add_argument( "--cnn2d-batch-norm", type=str, default="batch_norm",
            choices=["batch_norm", "none"],
        )

        group.add_argument( "--cnn2d-kernel-size", type=int, default=5)
        group.add_argument( "--cnn2d-padding", type=int, default=None)
        group.add_argument( "--cnn2d-classification-n-neurons", type=int, nargs="+", default=[64, 32])
        group.add_argument( "--cnn2d-n-targets", type=int, default=2)
        group.add_argument( "--cnn2d-classification-dropout", type=float, default=0.0)

        group.add_argument( "--cnn2d-classification-norm", type=str, default="batch_norm",
            choices=["batch_norm", "none"],
        )

        group.add_argument( "--cnn2d-classification-hidden-activation", type=str, default=None,
            choices=["relu", "leaky_relu", "gelu", "tanh"],
        )

        group.add_argument( "--cnn2d-classification-output-activation", type=str, default="sigmoid",
            choices=["sigmoid", "softmax", "none"],
        )

        group.add_argument( "--cnn2d-lr", type=float, default=1e-3)
        group.add_argument( "--cnn2d-weight-decay", type=float, default=1e-4)

        return group

    @staticmethod
    def from_args(args: argparse.Namespace, dm: ml_core.BaseDataModule) -> "CNN2D":
        """Create a CNN2D from command-line arguments."""

        activations = {
            "relu": torch.nn.ReLU,
            "leaky_relu": torch.nn.LeakyReLU,
            "gelu": torch.nn.GELU,
            "tanh": torch.nn.Tanh,
        }

        pooling = {
            "max_pool": torch.nn.MaxPool2d,
            "avg_pool": torch.nn.AvgPool2d,
            "none": None,
        }

        norm = {
            "batch_norm": torch.nn.BatchNorm2d,
            "none": None,
        }

        classification_norm = {
            "batch_norm": torch.nn.BatchNorm1d,
            "none": None,
        }

        output_activation = {
            "sigmoid": torch.nn.Sigmoid,
            "softmax": torch.nn.Softmax,
            "none": None,
        }

        hidden_activation = (
            activations[args.cnn2d_classification_hidden_activation]
            if args.cnn2d_classification_hidden_activation is not None
            else None
        )

        return CNN2D(
            input_shape=dm.get_sample_shape(),
            n_targets=dm.get_n_targets(),
            conv_n_neurons=args.cnn2d_conv_n_neurons,
            conv_activation=activations[args.cnn2d_conv_activation],
            conv_pooling=pooling[args.cnn2d_conv_pooling],
            conv_pooling_size=args.cnn2d_conv_pooling_size,
            conv_dilation=args.cnn2d_conv_dilation,
            conv_dropout=args.cnn2d_conv_dropout,
            batch_norm=norm[args.cnn2d_batch_norm],
            kernel_size=args.cnn2d_kernel_size,
            padding=args.cnn2d_padding,
            classification_n_neurons=args.cnn2d_classification_n_neurons,
            classification_dropout=args.cnn2d_classification_dropout,
            classification_norm=classification_norm[
                args.cnn2d_classification_norm
            ],
            classification_hidden_activation=hidden_activation,
            classification_output_activation=output_activation[
                args.cnn2d_classification_output_activation
            ],
            lr=args.cnn2d_lr,
            weight_decay=args.cnn2d_weight_decay,
        )
