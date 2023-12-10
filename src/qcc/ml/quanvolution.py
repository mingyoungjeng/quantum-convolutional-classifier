from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F

import pennylane as qml
from pennylane.templates import AngleEmbedding, RandomLayers

from qcc.filters import update_dims
from qcc.ml import reset_parameter
from qcc.ml.cnn import ConvolutionalNeuralNetwork, Layer

if TYPE_CHECKING:
    from typing import Optional


class Quanvolution(Module):
    __slots__ = (
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "padding_mode",
        "num_layers",
        "qnode",
        "seed",
        "params",
    )

    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]

    # padding_mode: str
    out_channels: Optional[int]

    num_layers: int
    qnode: Module
    seed: int

    def __init__(
        self,
        kernel_size: int | Iterable[int] = 2,
        stride: int | Iterable[int] = 1,
        padding: int | Iterable[int] = 0,
        dilation: int | Iterable[int] = 1,
        out_channels: Optional[int] = None,
        # padding_mode: str = "constant",
        num_layers: int = 4,
        parameterized: bool = True,
        **_,
    ) -> None:
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, Iterable)
            else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, Iterable) else (stride, stride)
        self.padding = padding if isinstance(padding, Iterable) else (padding, padding)
        self.dilation = (
            dilation if isinstance(dilation, Iterable) else (dilation, dilation)
        )
        # self.padding_mode = padding_mode
        self.out_channels = out_channels
        self.num_layers = num_layers

        rng = np.random.default_rng()
        self.seed = rng.integers(1000000)

        num_wires = np.prod(self.kernel_size)

        device = qml.device("default.qubit", wires=range(num_wires))
        qnode = qml.QNode(self.circuit, device, interface="torch")

        weight_shape = (self.num_layers, num_wires)
        self.qnode = qml.qnn.TorchLayer(
            qnode, {"params": weight_shape if parameterized else 0}
        )
        if not parameterized:
            self.params = torch.randint(low=0, high=1, size=weight_shape)

    def forward(self, input):
        input_shape = input.shape

        input = F.unfold(
            input,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )

        # Move axis with windows to end
        input = input.moveaxis(-1, -2)
        unfold_shape = input.shape

        # Reshape to (batches, windows)
        qnode_shape = (
            np.prod(unfold_shape[:-1]) * input_shape[1],
            unfold_shape[-1] // input_shape[1],
        )
        input = input.reshape(qnode_shape)

        # Quanvolution
        output: torch.Tensor = self.qnode(input)

        # Normalize back to 0-1
        # output = (output + 1) / 2

        output = output.reshape(unfold_shape)

        # Deal with output channels
        if self.out_channels is not None:
            output = F.adaptive_max_pool1d(output, self.out_channels)

        output = output.moveaxis(-2, -1)

        # Generate output shape
        output_shape = update_dims(
            input_shape[-2:],
            padding=self.padding,
            dilation=self.dilation,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        output_shape = (*output.shape[:-1], *output_shape)
        output = output.reshape(output_shape)

        return output

    def circuit(self, inputs: torch.Tensor, params: torch.Tensor):
        if len(params) == 0:
            params = self.params

        wires = range(params.shape[-1])

        # Normalize the input from [0, 1] to [0, π]
        inputs = torch.pi * inputs

        # Circuit
        AngleEmbedding(inputs, wires, rotation="Y")
        RandomLayers(params, wires, seed=self.seed)
        return [qml.expval(qml.PauliZ(j)) for j in wires]

    def reset_parameters(self):
        for parameter in self.parameters():
            reset_parameter(parameter)


class QuanvolutionalNeuralNetwork(torch.nn.Sequential):
    """
    Original QuanvolutionalNeuralNetwork as proposed
    a CNN with a pre-pended quanvolutional layer
    """

    quanvolution: Layer = Layer(Quanvolution, stride=2)

    def __init__(
        self,
        dims,
        num_layers: int = 1,
        num_features: int = 1,
        num_classes: int = 2,
        relu: bool = True,
        bias: bool = True,
        convolution: Module = None,
        pooling: Module = None,
        parameterized=True,
    ):
        num_channels = dims[2] if len(dims) > 2 else 1
        print(dims, num_channels)

        q = self.quanvolution(padding_mode="circular", parameterized=parameterized)
        dims = self.quanvolution.update_dims(*dims)
        dims = *dims[:2], 4 * num_channels
        num_layers += -1

        cnn = ConvolutionalNeuralNetwork(
            dims,
            num_layers,
            num_features,
            num_classes,
            relu=relu,
            bias=bias,
            convolution=convolution,
            pooling=pooling,
        )

        return super().__init__(q, cnn)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class QuanvolutionExtended(ConvolutionalNeuralNetwork):
    """
    All convolution layers in CNN are now quanvolution
    """

    convolution: Layer = Layer(Quanvolution, padding=1)
