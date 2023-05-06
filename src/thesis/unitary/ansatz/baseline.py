from itertools import zip_longest, tee
import numpy as np
from pennylane.operation import Operation
from thesis.unitary.baseline import BaselineConvolution, BaselinePooling1
from thesis.unitary.ansatz import Ansatz


# TODO: work with num_classes > 2
class BaselineAnsatz(Ansatz):
    convolve: type[Operation] = BaselineConvolution
    pool: type[Operation] = BaselinePooling1

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
            self.convolve(params, wires=wires)

    def _pooling(self, params, iterable):
        measurements = iterable[1::2]
        controlled = iterable[0::2]
        for wires in zip(measurements, controlled):
            self.pool(params, wires=wires)

        return controlled

    def __call__(self, params, *_, num_layers: int = None, **__):
        if num_layers is None:
            num_layers = self.max_layers

        wires = list(range(self.num_qubits))
        idx = np.cumsum([self.convolve.shape(), self.pool.shape()])
        conv_params, pool_params, params = np.split(params, idx)
        for i in range(num_layers):
            self._convolution(conv_params, wires)
            wires = self._pooling(pool_params, wires)

        return wires  # np.array(wires)  # .item()

    def shape(self, num_layers=None):
        if num_layers is None:
            num_layers = self.max_layers
        return (self.convolve.shape() + self.pool.shape()) * num_layers

    @property
    def max_layers(self, *_, **__) -> int:
        return int(np.ceil(np.log2(self.num_qubits)))
