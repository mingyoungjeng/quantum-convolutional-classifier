"""
https://github.com/takh04/QCNN
"""

from itertools import zip_longest, tee
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from qcnn.quantum.operation import Unitary
from qcnn.quantum.operation.ansatz import Ansatz


class BaselineFiltering(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        return [
            qml.U3(params[0], params[1], params[2], wires=wires[0]),
            qml.U3(params[3], params[4], params[5], wires=wires[1]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.RY(params[6], wires=wires[0]),
            qml.RZ(params[7], wires=wires[1]),
            qml.CNOT(wires=[wires[1], wires[0]]),
            qml.RY(params[8], wires=wires[0]),
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.U3(params[9], params[10], params[11], wires=wires[0]),
            qml.U3(params[12], params[13], params[14], wires=wires[1]),
        ]

    @staticmethod
    def _shape(*_) -> int:
        return 15


class BaselinePooling1(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        return [
            qml.CRZ(params[0], wires=[wires[0], wires[1]]),
            qml.PauliX(wires=wires[0]),
            qml.CRX(params[1], wires=[wires[0], wires[1]]),
        ]

    @staticmethod
    def _shape(*_) -> int:
        return 2


class BaselinePooling2(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        return [
            qml.CRZ(params[0], wires=[wires[0], wires[1]]),
            qml.PauliX(wires=wires[0]),
            qml.CRX(params[1], wires=[wires[0], wires[1]]),
            qml.PauliX(wires=wires[0]),
        ]

    @staticmethod
    def _shape(*_) -> int:
        return 2


class BaselinePooling3(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        m_0 = qml.measure(wires[0])
        qml.cond(m_0 == 0, qml.RY)(params[0], wires=wires[1])
        qml.cond(m_0 == 1, qml.RY)(params[1], wires=wires[1])

    @staticmethod
    def _shape(*_) -> int:
        return 2


# TODO: work with num_classes > 2
class BaselineAnsatz(Ansatz):
    convolve: type[Operation] = BaselineFiltering
    pool: type[Operation] = BaselinePooling1

    @classmethod
    def _convolution(cls, params, iterable):
        a, b = tee(iterable)
        first = next(b, None)
        lst = list(zip_longest(a, b, fillvalue=first))
        last = lst.pop()[::-1]
        lst = lst[::2] + lst[1::2]

        if len(lst) > 1:
            lst.insert(0, last)

        for wires in lst:
            cls.convolve(params, wires=wires)

    @classmethod
    def _pooling(cls, params, iterable):
        measurements = iterable[1::2]
        controlled = iterable[0::2]

        for wires in zip(measurements, controlled):
            cls.pool(params, wires=wires)

        return controlled

    def circuit(self, params):
        idx = np.cumsum([self.convolve.shape(), self.pool.shape()])
        wires = self.qubits.flatten()
        for _ in range(self.num_layers):
            conv_params, pool_params, params = np.split(params, idx)
            self._convolution(conv_params, wires)
            wires = self._pooling(pool_params, wires)

        return wires

    @property
    def shape(self):
        return (self.convolve.shape() + self.pool.shape()) * self.num_layers

    @property
    def max_layers(self) -> int:
        return int(np.ceil(np.log2(self.qubits.total)))
