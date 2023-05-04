from unitary.baseline import (
    BaselineConvolution,
    BaselinePooling1,
    BaselinePooling2,
    BaselinePooling3,
)
from ansatz.ansatz import ConvolutionAnsatz
from itertools import zip_longest, tee
import numpy as np


# TODO: work with num_classes > 2
class BaselineAnsatz(ConvolutionAnsatz):
    convolve = BaselineConvolution()
    pool = BaselinePooling1()

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def _convolution(self, params, iterable):
        a, b = tee(iterable)
        first = next(b, None)
        lst = list(zip_longest(a, b, fillvalue=first))
        last = lst.pop()[::-1]
        lst = lst[::2] + lst[1::2]
        lst.insert(0, last)

        for wires in lst:
            p = self.convolve(params, wires=wires)

        return p

    def _pooling(self, params, iterable):
        measurements = iterable[1::2]
        controlled = iterable[0::2]
        for wires in zip(measurements, controlled):
            p = self.pool(params, wires=wires)

        return controlled, p

    def __call__(self, params, *_, num_layers: int = None, **__):
        if num_layers is None:
            num_layers = self.max_layers

        wires = list(range(self.num_qubits))
        for i in range(num_layers):
            params = self._convolution(params, wires)
            wires, params = self._pooling(params, wires)

        return wires  # np.array(wires)  # .item()

    def total_params(self, num_layers=None, *_, **__):
        if num_layers is None:
            num_layers = self.max_layers

        return (self.convolve.total_params() + self.pool.total_params()) * num_layers

    @property
    def max_layers(self) -> int:
        return int(np.ceil(np.log2(self.num_qubits)))
