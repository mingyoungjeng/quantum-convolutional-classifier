from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from attrs import define, field, asdict, Factory
from torch import nn
import numpy as np
from qcnn.ml import USE_CUDA
from qcnn.ml.model import Model

if TYPE_CHECKING:
    # from typing import Iterable
    from torch.nn import Module


@define  # (slots=False)
class Layer:
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

    def __call__(self, module: type[Module], *args, **kwargs) -> Module:
        kwargs.update(asdict(self))
        return module(*args, **kwargs)

    def update_dims(self, *dims):
        params = asdict(self)
        for key, value in params.items():
            if not isinstance(value, Iterable):
                params[key] = [value] * len(dims)
        params["size"] = dims

        new_dims = tuple(
            self._size(**dict(zip(params, t))) for t in zip(*params.values())
        )
        return new_dims


@define
class CNN(Model):
    cnn: nn.Sequential = field(init=False)
    convolution: Layer = Layer()
    pooling: Layer = Layer()
    num_classes: int = Factory(lambda self: len(self.data.classes), takes_self=True)
    num_features = 3

    # Convolutional layer parameters

    # @cnn.default
    # def _default_cnn(self):
    # return nn.Sequential(
    #     nn.Conv2d(in_channels=1, out_channels=n_feature, kernel_size=2, padding=1),
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2),
    #     nn.Conv2d(
    #         in_channels=n_feature, out_channels=n_feature, kernel_size=2, padding=1
    #     ),
    #     nn.ReLU(),
    #     nn.MaxPool2d(kernel_size=2),
    #     nn.Flatten(),
    #     nn.Linear(n_feature * final_layer_size, num_classes),
    # )

    def forward(self, dims, num_layers=1):
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
                    nn.Conv2d,
                    in_channels=channels if i == 0 else self.num_features,
                    out_channels=self.num_features,
                )
            ]
            dims = self.convolution.update_dims(*dims)

            # Pooling
            lst += [self.pooling(nn.MaxPool2d)]
            dims = self.convolution.update_dims(*dims)

            # ReLU
            lst += [nn.ReLU()]

        lst += [
            nn.Flatten(),
            nn.Linear(self.num_features * np.prod(dims), self.num_classes),
        ]

        module = nn.Sequential(*lst)
        return module.cuda() if USE_CUDA else module

    def __call__(self, dims, num_layers=1, silent=False):
        model = self.forward(dims, num_layers)
        return super().__call__(model, silent=silent)
