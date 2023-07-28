from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F

import pennylane as qml
from pennylane.templates import AngleEmbedding, RandomLayers

from qcc.ml import reset_parameter
from qcc.ml.cnn import ConvolutionalNeuralNetwork, Layer


class Quanvolution(Module):
    __slots__ = (
        "kernel_size",
        "stride",
        "padding",
        "dilation",
        "padding_mode",
        "num_layers",
        "qnode",
    )

    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]

    # padding_mode: str
    out_channels: int

    num_layers: int
    qnode: Module

    def __init__(
        self,
        kernel_size: int | Iterable[int] = 2,
        stride: int | Iterable[int] = 1,
        padding: int | Iterable[int] = 0,
        dilation: int | Iterable[int] = 1,
        out_channels: int = 1,
        # padding_mode: str = "constant",
        num_layers: int = 4,
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

        num_wires = self.kernel_size[0] * self.kernel_size[1]

        device = qml.device("default.qubit", wires=range(num_wires))
        qnode = qml.QNode(self.circuit, device, interface="torch")
        self.qnode = qml.qnn.TorchLayer(qnode, {"params": (self.num_layers, num_wires)})

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
        output = (output + 1) / 2

        # Deal with output channels and undo shape
        output = output.reshape(unfold_shape)
        output = F.adaptive_max_pool1d(output, self.out_channels)
        output = output.moveaxis(-2, -1)

        # Generate output shape
        output_shape = (
            Layer._size(*args)
            for args in zip(
                input_shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
            )
        )
        output_shape = (*output.shape[:-1], *output_shape)
        output = output.reshape(output_shape)

        return output

    @staticmethod
    def circuit(inputs: torch.Tensor, params: torch.Tensor):
        wires = range(params.shape[-1])

        # Normalize the input from [0, 1] to [0, π]
        inputs = torch.pi * inputs

        # Circuit
        AngleEmbedding(inputs, wires, rotation="Y")
        RandomLayers(params, wires)
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

    def __init__(self, dims, num_layers=1, num_features=1, num_classes=2):
        q = self.quanvolution(out_channels=num_features, padding_mode="circular")
        dims = self.quanvolution.update_dims(*dims)
        dims = (*dims, num_features)
        num_layers += -1

        cnn = ConvolutionalNeuralNetwork(dims, num_layers, num_features, num_classes)

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
