from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from attrs import define, asdict, filters
from torch import nn
import numpy as np

from qcc.filters import update_dims

if TYPE_CHECKING:
    from torch.nn import Module


@define
class Layer:
    """Wrapper of convolution/pooling layers that stores parameters and tracks data size"""

    module: type[Module]
    kernel_size: int | Iterable[int] = 2
    stride: int | Iterable[int] = 1
    padding: int | Iterable[int] = 0
    dilation: int | Iterable[int] = 1

    def __call__(self, *args, **kwargs) -> Module:
        kwargs.update(self.params())

        if issubclass(self.module, nn.modules.pooling._AvgPoolNd):
            del kwargs["dilation"]

        return self.module(*args, **kwargs)

    def update_dims(self, *dims):
        return update_dims(dims, **self.params())

    def params(self):
        attributes = ["kernel_size", "stride", "padding", "dilation"]
        return asdict(self, filter=filters.include(*attributes))


class ConvolutionalNeuralNetwork(nn.Sequential):
    """Just a normal CNN"""

    def __init__(
        self,
        dims: Iterable[int],
        num_layers: int = 1,
        num_features: int = 1,
        num_classes: int = 2,
        relu: bool = True,
        bias: bool = True,
        convolution: nn.Module | None = None,
        pooling: nn.Module | None = None,
    ):
        """
        Args:
            dims (Iterable[int]): Dimensions of data in column-major order (breaking from PyTorch expectations). Ex: [16, 16, 3] not [3, 16, 16].
            num_layers (int): Number of convolution-pooling layers. Defaults to 1.
            num_features (int): Number of features per convolution layer. Defaults to 1.
            num_classes (int): Number of output clases. Defaults to 2.
            relu (bool): Whether to include ReLU layers after each pooling layer. Defaults to True.
            bias (bool): Whether to include a bias term in convolution layer. Defaults to True.
            convolution (nn.Module): Fixed convolution layer / module. Defaults to None.
            pooling (nn.Module): Fixed pooling layer / module. Defaults to None.
        """
        *dims, channels = dims

        # Setup convolution layer
        if convolution is not None:
            conv_layer = Layer(convolution, padding=1)

        # Setup pooling layer
        if pooling is not None:
            pool_layer = Layer(pooling, stride=2)

        lst = []
        for i in range(num_layers):
            # ==== convolution layer ==== #
            if convolution is None:
                conv_layer = _guess_convolution(*dims)
            lst += [
                conv_layer(
                    in_channels=channels if i == 0 else num_features,
                    out_channels=num_features,
                    padding_mode="circular",
                    bias=bias,
                )
            ]
            dims = conv_layer.update_dims(*dims)

            # ==== pooling layer==== #
            if pooling is None:
                pool_layer = _guess_pooling_max(*dims)
            lst += [pool_layer()]
            dims = pool_layer.update_dims(*dims)

            # ==== ReLU layer ==== #
            if relu:
                lst += [nn.ReLU()]

        # ==== fully-connected layer ==== #
        lst += [
            nn.Flatten(),
            nn.Linear(num_features * np.prod(dims, dtype=int), num_classes),
        ]

        super().__init__(*lst)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


# ==== TODO: eventually optimize and remove these because they are ugly and clunky ==== #


def _guess_convolution(*dims: int) -> Layer:
    """Guess convolution layer if not defined"""
    match len(dims):
        case 3:
            module = nn.Conv3d
        case 2:
            module = nn.Conv2d
        case _:
            module = nn.Conv1d

    return Layer(module, padding=1)


def _guess_pooling_max(*dims: int) -> Layer:
    """Guess max pooling layer if not defined"""
    match len(dims):
        case 3:
            module = nn.MaxPool3d
        case 2:
            module = nn.MaxPool2d
        case _:
            module = nn.MaxPool1d

    return Layer(module, stride=2)


def _guess_pooling_avg(*dims: int) -> Layer:
    """Guess avg pooling layer if not defined"""
    match len(dims):
        case 3:
            module = nn.AvgPool3d
        case 2:
            module = nn.AvgPool2d
        case _:
            module = nn.AvgPool1d

    return Layer(module, stride=2)
