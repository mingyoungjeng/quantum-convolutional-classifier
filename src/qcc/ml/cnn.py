from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from attrs import define, asdict, filters
from torch import nn
import numpy as np

from qcc.filters import update_dims

if TYPE_CHECKING:
    from torch.nn import Module
    from typing import Optional


@define
class Layer:
    """Wrapper of convolution/pooling layers that stores parameters and tracks data size."""

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
        dims: list[int],
        num_layers: int = 1,
        num_features: int = 1,
        num_classes: int = 2,
        relu: bool = True,
        bias: bool = True,
        convolution: Optional[nn.Module] = None,
        pooling: Optional[nn.Module] = None,
    ):
        """
        Args:
            dims (list[int]): _description_
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
                # Guess convolution layer if not defined
                match len(dims):
                    case 3:
                        conv_layer = Layer(nn.Conv3d, padding=1)
                    case 2:
                        conv_layer = Layer(nn.Conv2d, padding=1)
                    case _:
                        conv_layer = Layer(nn.Conv1d, padding=1)
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
                # Guess pooling layer if not defined
                match len(dims):
                    case 3:
                        pool_layer = Layer(nn.MaxPool3d, stride=2)
                    case 2:
                        pool_layer = Layer(nn.MaxPool2d, stride=2)
                    case _:
                        pool_layer = Layer(nn.MaxPool1d, stride=2)
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
