from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, Optional

from functools import partial

from torch import nn
from torch.nn import Module
from torch.linalg import norm

import numpy as np

from qcc.filters import update_dims
from qcc.ml import init_params
from qcc.ml.cnn import Layer
from qcc.quantum import reconstruct
from qcc.quantum.pennylane import Unitary

from qcc.quantum.pennylane.ansatz import MQCC
from qcc.quantum.pennylane.local import define_filter
from qcc.quantum.pennylane.c2q import (
    ConvolutionAngleFilter,
    ConvolutionComplexAngleFilter,
    ConvolutionFilter,
)
from qcc.quantum.pennylane.ansatz import FullyConnected

if TYPE_CHECKING:
    from typing import Optional
    from qcc.quantum.pennylane import Unitary

AnsatzFilter = define_filter(num_layers=4)


class MQCCHybrid(nn.Sequential):
    def __init__(
        self,
        dims,
        num_layers: int = 1,
        num_features: int = 1,
        num_classes: int = 2,
        relu: bool = True,
        bias: bool = False,
        U_filter: type[Unitary] = ConvolutionAngleFilter,
        U_fully_connected: Optional[type[Unitary]] = None,
    ):
        *dims, channels = dims

        layer = Layer(MQCCLayer, padding=1)

        lst = []
        for i in range(num_layers):
            # Convolution + Pooling
            mqcc: MQCCLayer = layer(
                dims,
                in_channels=channels if i == 0 else num_features,
                out_channels=num_features,
                pooling=True,
                U_filter=U_filter,
                bias=bias,
            )

            lst += [mqcc]
            dims = mqcc.update_dims(dims)

            # ReLU
            if relu:
                lst += [nn.ReLU()]

        if U_fully_connected is None:
            fully_connected = nn.Linear
        else:
            fully_connected = partial(FullyConnectedLayer, U_filter=U_fully_connected)

        lst += [
            nn.Flatten(),
            fully_connected(num_features * np.prod(dims, dtype=int), num_classes),
        ]

        super().__init__(*lst)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


# TODO: rename
class MQCCNonHybrid(MQCCHybrid):
    """Quantum-only"""

    def __init__(
        self,
        dims,
        num_layers: int = 1,
        num_features: int = 1,
        num_classes: int = 2,
        U_filter: type[Unitary] = ConvolutionAngleFilter,
        U_fully_connected: type[Unitary] = ConvolutionAngleFilter,
    ):
        super().__init__(
            dims,
            num_layers,
            num_features,
            num_classes,
            relu=False,
            bias=False,
            U_filter=U_filter,
            U_fully_connected=U_fully_connected,
        )


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
        bias: bool = False,
        U_filter: type[Unitary] = ConvolutionAngleFilter,
    ):
        super().__init__()

        self.dims = dims
        self.in_channels = in_channels
        self.out_channels = out_channels

        names = ["kernel_size", "stride", "padding", "dilation"]
        params = [kernel_size, stride, padding, dilation]
        for name, param in zip(names, params):
            param = [param] * len(dims) if isinstance(param, int) else param
            setattr(self, name, param)

        num_features = int(np.ceil(out_channels / in_channels))

        module_options = {
            "U_filter": U_filter,
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

        if bias:
            self.register_parameter("bias", init_params(self.out_channels))

        self.register_parameter("filter_norm", init_params(self.out_channels))
        self.reset_parameters()

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

        try:  # Apply bias term(s) (if possible)
            bias = self.get_parameter("bias")
            bias = bias.reshape(self.out_channels, *(1 for _ in range(result.ndim - 1)))

            result = result + bias
        except AttributeError:
            pass

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
        try:  # Reset bias (if possible)
            k = self.in_channels * np.prod(self.kernel_size)
            k = np.sqrt(1 / k)
            nn.init.uniform_(self.get_parameter("bias"), -k, k)
        except AttributeError:
            pass

        # Trainable parameter for the magnitude of filter(s)
        nn.init.uniform_(self.get_parameter("filter_norm"), -1, 1)

        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class FullyConnectedLayer(Module):
    """Wrapper for PyTorch compatibility"""

    __slots__ = ("in_features", "out_features")

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        U_filter: type[Unitary] = ConvolutionAngleFilter,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mqcc = FullyConnected.from_dims(
            [in_features],
            out_features,
            U_filter=U_filter,
        )

        if bias:
            self.register_parameter("bias", init_params(self.out_features))

        self.register_parameter("norm", init_params(self.out_features))
        self.reset_parameters()

    def forward(self, inputs):
        # Normalize inputs
        magnitudes = norm(inputs, dim=1)
        inputs = (inputs.T / magnitudes).T

        result = self.mqcc.forward(inputs)

        # Unnormalize output
        result = (result.T / magnitudes).T

        try:  # Apply bias term(s) (if possible)
            bias = self.get_parameter("bias").unsqueeze(0)

            result = result + bias
        except AttributeError:
            pass

        return result.float()

    def reset_parameters(self):
        try:  # Reset bias (if possible)
            k = np.sqrt(1 / self.in_features)
            nn.init.uniform_(self.get_parameter("bias"), -k, k)
        except AttributeError:
            pass

        # Magnitude of inverse-c2q parameters
        nn.init.uniform_(self.get_parameter("norm"), -1, 1)

        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
