"""
https://github.com/takh04/QCNN
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import count, zip_longest, tee

import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.parametervector import ParameterVector

from qcc.quantum.qiskit.ansatz import Ansatz

if TYPE_CHECKING:
    from typing import Optional, Iterable


class QCNNFilter(Gate):
    instances_counter = count()

    @property
    def _id(self):
        i = next(self.instances_counter)
        return str() if i == 0 else i

    def __init__(self, label: Optional[str] = None) -> None:
        params_name = f"qcnn_filter{self._id}"
        params = ParameterVector(params_name, length=15)

        super().__init__("QCNN Filter", 2, params, label)

    def _define(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        params = self.params
        qubits = qc.qubits

        qc.u(params[0], params[1], params[2], qubit=qubits[0])
        qc.u(params[3], params[4], params[5], qubit=qubits[1])
        qc.cnot(qubits[0], qubits[1])
        qc.ry(params[6], qubit=qubits[0])
        qc.rz(params[7], qubit=qubits[1])
        qc.cnot(qubits[1], qubits[0])
        qc.ry(params[8], qubit=qubits[0])
        qc.cnot(qubits[0], qubits[1])
        qc.u(params[9], params[10], params[11], qubit=qubits[0])
        qc.u(params[12], params[13], params[14], qubit=qubits[1])

        self.definition = qc


class QCNNPooling(Gate):
    instances_counter = count()

    @property
    def _id(self):
        i = next(self.instances_counter)
        return str() if i == 0 else i

    def __init__(self, label: Optional[str] = None) -> None:
        params_name = f"qcnn_pooling{self._id}"
        params = ParameterVector(params_name, length=2)

        super().__init__("QCNN Pooling", 2, params, label)

    def _define(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        params = self.params
        qubits = qc.qubits

        qc.crz(params[0], qubits[0], qubits[1])
        qc.x(qubit=qubits[0])
        qc.crx(params[1], qubits[0], qubits[1])

        self.definition = qc


class QCNN(Ansatz):
    __slots__ = "convolve", "pool"
    convolve: type[Gate]
    pool: type[Gate]

    def __init__(
        self,
        qubits,
        num_layers: int = 1,
        convolve=QCNNFilter,
        pool=QCNNPooling,
    ):
        self.convolve = convolve
        self.pool = pool
        super().__init__(qubits, num_layers)

    def _convolution(self, qc: QuantumCircuit, iterable: Iterable):
        a, b = tee(iterable)
        first = next(b, None)
        lst = list(zip_longest(a, b, fillvalue=first))
        last = lst.pop()[::-1]
        lst = lst[::2] + lst[1::2]

        if len(lst) > 1:
            lst.insert(0, last)

        gate = self.convolve()
        for qubits in lst:
            qc.compose(gate, qubits=qubits, inplace=True)

    def _pooling(self, qc, iterable: Iterable):
        measurements = iterable[1::2]
        controlled = iterable[0::2]

        gate = self.pool()
        for qubits in zip(measurements, controlled):
            qc.compose(gate, qubits=qubits, inplace=True)

        return controlled

    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(len(self.qubits.flatten()), name=type(self).__name__)

        qubits = qc.qubits
        for _ in range(self.num_layers):
            self._convolution(qc, qubits)
            qubits = self._pooling(qc, qubits)

        meas = {qc.find_bit(q)[0] for q in qubits}
        return qc, meas

    @property
    def max_layers(self) -> int:
        return int(np.ceil(np.log2(self.qubits.total)))
