from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from attrs import define, asdict, filters
from torch import nn
import numpy as np

if TYPE_CHECKING:
    from torch.nn import Module


@define
class Layer:
    module: type[Module]
    kernel_size: int | Iterable[int] = 2
    stride: int | Iterable[int] = 1
    padding: int | Iterable[int] = 0
    dilation: int | Iterable[int] = 1

    @staticmethod
    def _size(
        size: int,
        padding: int,
        dilation: int,
        kernel_size: int,
        stride: int,
    ) -> int:
        size += 2 * padding
        size += -dilation * (kernel_size - 1)
        size += -1
        size = size // stride
        size += 1

        return size

    def __call__(self, *args, **kwargs) -> Module:
        kwargs.update(self.params())

        if issubclass(self.module, nn.modules.pooling._AvgPoolNd):
            del kwargs["dilation"]

        return self.module(*args, **kwargs)

    def update_dims(self, *dims):
        params = self.params()
        for key, value in params.items():
            if not isinstance(value, Iterable):
                params[key] = [value] * len(dims)
        params["size"] = dims

        new_dims = tuple(
            self._size(**dict(zip(params, t))) for t in zip(*params.values())
        )
        return new_dims

    def params(self):
        return asdict(self, filter=filters.exclude("module"))


class ConvolutionalNeuralNetwork(nn.Sequential):
    convolution: Layer = Layer(nn.Conv2d, padding=1)
    pooling: Layer = Layer(nn.AvgPool2d, stride=2)

    def __init__(self, dims, num_layers=1, num_features=1, num_classes=2,):
        if len(dims) > 2:
            width, height, channels, *_ = dims
            dims = width, height
        else:
            channels = 1

        lst = []
        for i in range(num_layers):
            # Convolution
            lst += [
                self.convolution(
                    in_channels=channels if i == 0 else num_features,
                    out_channels=num_features,
                    padding_mode="circular",
                )
            ]
            dims = self.convolution.update_dims(*dims)

            # Pooling
            lst += [self.pooling()]
            dims = self.pooling.update_dims(*dims)

            # ReLU
            # lst += [nn.ReLU()]

        lst += [
            nn.Flatten(),
            nn.Linear(num_features * np.prod(dims), num_classes),
        ]

        super().__init__(*lst)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
