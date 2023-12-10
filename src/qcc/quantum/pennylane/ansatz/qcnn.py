"""
https://github.com/takh04/QCNN
"""

from itertools import zip_longest, tee
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from qcc.quantum.pennylane import Unitary
from qcc.quantum.pennylane.ansatz import Ansatz


class QCNNConvolution(Unitary):
    """QCNN Convolution Ansatz"""

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
    def _shape(*_, **__) -> int:
        return 15


class QCNNPooling1(Unitary):
    """QCNN Pooling Ansatz from Related Work"""

    @staticmethod
    def compute_decomposition(params, wires, **_):
        return [
            qml.CRZ(params[0], wires=[wires[0], wires[1]]),
            qml.PauliX(wires=wires[0]),
            qml.CRX(params[1], wires=[wires[0], wires[1]]),
        ]

    @staticmethod
    def _shape(*_, **__) -> int:
        return 2


class QCNNPooling2(Unitary):
    """My Experimental QCNN Pooling Ansatz"""

    @staticmethod
    def compute_decomposition(params, wires, **_):
        return [
            qml.CRZ(params[0], wires=[wires[0], wires[1]]),
            qml.PauliX(wires=wires[0]),
            qml.CRX(params[1], wires=[wires[0], wires[1]]),
            qml.PauliX(wires=wires[0]),
        ]

    @staticmethod
    def _shape(*_, **__) -> int:
        return 2


class QCNNPooling3(Unitary):
    """My Experimental QCNN Pooling Ansatz"""

    @staticmethod
    def compute_decomposition(params, wires, **_):
        m_0 = qml.measure(wires[0])
        qml.cond(m_0 == 0, qml.RY)(params[0], wires=wires[1])
        qml.cond(m_0 == 1, qml.RY)(params[1], wires=wires[1])

    @staticmethod
    def _shape(*_, **__) -> int:
        return 2


# TODO: work with num_classes > 2
class QCNN(Ansatz):
    """Quantum Convolutional Neural Network (Cong, 2018)"""

    __slots__ = "convolve", "pool"

    convolve: type[Operation]
    pool: type[Operation]

    def __init__(
        self,
        qubits,
        num_layers: int = 1,
        convolve=QCNNConvolution,
        pool=QCNNPooling1,
    ):
        self.convolve = convolve
        self.pool = pool
        super().__init__(qubits, num_layers)

    def _convolution(self, params, iterable):
        a, b = tee(iterable)
        first = next(b, None)
        lst = list(zip_longest(a, b, fillvalue=first))
        last = lst.pop()[::-1]
        lst = lst[::2] + lst[1::2]

        if len(lst) > 1:
            lst.insert(0, last)

        for wires in lst:
            self.convolve(params, wires=wires)

    def _pooling(self, params, iterable):
        measurements = iterable[1::2]
        controlled = iterable[0::2]

        for wires in zip(measurements, controlled):
            self.pool(params, wires=wires)

        return controlled

    def circuit(self, *params):
        (params,) = params
        idx = np.cumsum([self.convolve.shape(), self.pool.shape()])
        wires = self.qubits.flatten()
        for _ in range(self.num_layers):
            conv_params, pool_params, params = np.split(params, idx)
            self._convolution(conv_params, wires)
            wires = self._pooling(pool_params, wires)

        return wires

    @property
    def shape(self):
        return (self.convolve.shape(2) + self.pool.shape(2)) * self.num_layers

    @property
    def max_layers(self) -> int:
        return int(np.ceil(np.log2(self.qubits.total)))
