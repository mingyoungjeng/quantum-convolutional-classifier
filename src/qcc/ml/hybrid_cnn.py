from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from torch import nn
from torch.nn import Module
from torch.linalg import norm

import numpy as np

from qcc.filters import update_dims
from qcc.ml.cnn import Layer
from qcc.quantum import to_qubits, reconstruct

from qcc.quantum.pennylane.ansatz import MQCC
from qcc.quantum.pennylane.c2q import (
    ConvolutionAngleFilter,
    ConvolutionComplexAngleFilter,
    ConvolutionFilter,
)

if TYPE_CHECKING:
    pass


class MQCCLayer(Module):
    __slots__ = (
        "dims",
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "padding",
        "dilation",
    )

    def __init__(
        self,
        dims,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Iterable[int],
        stride: int | Iterable[int],
        padding: int | Iterable[int],
        dilation: int | Iterable[int],
        pooling: bool = False,
    ):
        super().__init__()

        self.dims = dims
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = (
            [kernel_size] * len(dims) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = [stride] * len(dims) if isinstance(stride, int) else stride
        self.padding = [padding] * len(dims) if isinstance(padding, int) else padding
        self.dilation = (
            [dilation] * len(dims) if isinstance(dilation, int) else dilation
        )

        num_features = int(np.ceil(out_channels / in_channels))

        module_options = {
            "U_filter": ConvolutionAngleFilter,
            "num_features": num_features,
            "U_fully_connected": None,
            "pooling": pooling,
            "filter_shape": self.kernel_size,
        }

        self.mqcc = MQCC.from_dims(
            (*dims, in_channels),
            num_layers=1,
            **module_options,
        )

    def forward(self, inputs):
        """
        Expects inputs in form (batch_size, *row_major_dims)
        """

        dims = self.dims
        batch_size = inputs.shape[0]

        inputs = inputs.reshape((batch_size, self.in_channels * np.prod(dims)))

        # Normalize inputs
        magnitudes = norm(inputs, dim=1)
        inputs = (inputs.T / magnitudes).T

        result = self.mqcc.forward(inputs)

        # Unnormalize output
        result = (result.T / magnitudes).T

        # Column-major correction (batch_size)
        result = result.moveaxis(0, -1)

        dims_out = self.update_dims(dims)
        dims = update_dims(
            dims,
            kernel_size=self.mqcc.pooling,
            stride=self.mqcc.pooling,
        )

        dims_out = (batch_size, *dims_out, self.in_channels, self.mqcc.num_features)
        dims = (batch_size, *dims, self.in_channels, self.mqcc.num_features)

        result = reconstruct(result, dims, dims_out)
        result = result.T  # Return in row-major order

        # Merge features into least-significant dimension
        dims_out = (*dims_out[:-2], dims_out[-2] * dims_out[-1])
        result = result.reshape(dims_out[::-1])

        # Crop out_channels if necessary
        result = result[: self.out_channels, ...]

        # Column-major correction (batch_size)
        result = result.moveaxis(-1, 0)

        return result.float()

    def update_dims(self, dims):
        dims = update_dims(
            dims,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        # Correct for pooling
        dims = update_dims(
            dims,
            kernel_size=self.mqcc.pooling,
            stride=self.mqcc.pooling,
        )
        return dims

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class MQCCHybrid(nn.Sequential):
    def __init__(
        self,
        dims,
        num_layers: int = 1,
        num_features: int = 1,
        num_classes: int = 2,
        relu: bool = True,
    ):
        *dims, channels = dims

        layer = Layer(MQCCLayer, padding=1)

        lst = []
        for i in range(num_layers):
            # Convolution + Pooling
            mqcc = layer(
                dims,
                in_channels=channels if i == 0 else num_features,
                out_channels=num_features,
                pooling=True,
            )

            lst += [mqcc]
            dims = mqcc.update_dims(dims)

            # ReLU
            if relu:
                lst += [nn.ReLU()]

        lst += [
            nn.Flatten(),
            nn.Linear(num_features * np.prod(dims, dtype=int), num_classes),
        ]

        super().__init__(*lst)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
