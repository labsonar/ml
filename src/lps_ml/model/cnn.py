"""
Module containing a Convolutional Neural Network (CNN) based models.
"""
import typing
import torch

import lps_ml.model.mlp as lps_mlp
from lps_utils.log import warning

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
        lr: float = 1e-3,
        loss_fn: typing.Optional[typing.Callable[[], torch.nn.Module]] = None,
    ):
        conv_activation = conv_activation or torch.nn.ReLU
        conv_pooling = conv_pooling or torch.nn.MaxPool2d
        conv_pooling_size = conv_pooling_size or [2, 2]
        batch_norm = batch_norm or torch.nn.BatchNorm2d
        classification_norm = classification_norm or torch.nn.BatchNorm1d

        classification_hidden_activation = classification_hidden_activation or conv_activation
        padding = padding or int((kernel_size - 1) / 2)

        if len(input_shape) != 3:
            raise ValueError(f"CNN expects as input an image in the format: \
                                    channel x width x height (current {input_shape})")

        if isinstance(conv_dilation, int):
            conv_dilation = [conv_dilation] * len(conv_n_neurons)

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
            if conv_dropout != 0 and i != 0:
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
            loss_fn=loss_fn
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
        features = self.to_feature_space(inputs)
        out = super().forward(features)
        return out
