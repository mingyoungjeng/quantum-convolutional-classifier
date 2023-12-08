from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from itertools import count

import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.gate import Gate
from qiskit.circuit.parametervector import ParameterVector

from qcc.quantum import to_qubits
from qcc.quantum.qiskit.shift import Shift
from qcc.quantum.qiskit.c2q import C2Q

if TYPE_CHECKING:
    from typing import Optional


class Convolution(Gate):
    instances_counter = count()
    dims: Iterable[int]
    filter_shape: Iterable[int]

    @property
    def _id(self):
        i = next(self.instances_counter)
        return str() if i == 0 else i

    def __init__(
        self,
        dims: Iterable[int],
        filter_shape: Iterable[int] = (1,),
        label: Optional[str] = None,
    ) -> None:
        params_name = f"filter{self._id}"
        params = ParameterVector(params_name, length=np.prod(filter_shape))

        self.filter_shape = (
            filter_shape if isinstance(filter_shape, Iterable) else (filter_shape,)
        )
        self.dims = dims if isinstance(dims, Iterable) else (dims,)

        if len(self.filter_shape) > len(self.dims):
            msg = f"Dimensionality of filter larger than dimensionality of data ({len(self.filter_shape)} > {len(self.dims)})"
            raise ValueError(msg)

        num_qubits = sum(self.dims_q) + sum(self.filter_shape_q)
        super().__init__("convolution", int(num_qubits), params, label)

    @property
    def dim(self):
        return len(self.filter_shape)

    @property
    def dims_q(self):
        return to_qubits(self.dims)

    @property
    def filter_shape_q(self):
        return to_qubits(self.filter_shape)

    @staticmethod
    def shift(qc: QuantumCircuit, filter_shape_q, stride=1, H=True) -> QuantumCircuit:
        for i, fsq in enumerate(filter_shape_q):
            data_qubits = qc.qregs[i]
            kernel_qubits = qc.qregs[i - len(filter_shape_q)][:fsq]

            if len(data_qubits) == 0:
                continue

            # Apply Hadamard to filter qubits
            if H:
                qc.h(kernel_qubits)

            # Shift operation
            for j, control in enumerate(kernel_qubits):
                gate = Shift(k=-stride, num_qubits=len(data_qubits[j:])).control()
                qc.compose(gate, [control, *data_qubits[j:]], inplace=True)

        return qc

    @staticmethod
    def filter(qc: QuantumCircuit, fltr: np.ndarray) -> QuantumCircuit:
        qubits = [q[:fsq] for q, fsq in zip(qc.qregs, to_qubits(fltr.shape))]
        qubits = [x for q in qubits for x in q]

        # Create gate
        c2q = QuantumCircuit(len(qubits))
        c2q.compose(C2Q(len(qubits), inplace=True))

        # Mapping parameters
        fltr = np.conj(fltr).flatten(order="F")
        mapping = {key: value for key, value in zip(c2q.parameters, fltr)}
        c2q.assign_parameters(mapping, inplace=True)

        # Add inverse gate to QuantumCircuit
        qc.compose(c2q.inverse(), qubits, inplace=True)

        return qc

    @staticmethod
    def permute(qc: QuantumCircuit, filter_shape_q):
        for i, fsq in enumerate(filter_shape_q):
            data_qubits = qc.qregs[i][:fsq]
            kernel_qubits = qc.qregs[i - len(filter_shape_q)][:fsq]

            for f, a in zip(data_qubits, kernel_qubits):
                qc.swap(f, a)

        return qc

    @staticmethod
    def filter_post_permute(qc: QuantumCircuit, fltr: np.ndarray) -> QuantumCircuit:
        fltr = np.conj(fltr).flatten(order="F")
        num_qubits = int(to_qubits(len(fltr)))

        # Create gate
        c2q = QuantumCircuit(num_qubits)
        c2q.compose(C2Q(num_qubits), inplace=True)

        # Mapping parameters
        mapping = {key: value for key, value in zip(c2q.parameters, fltr)}
        c2q.assign_parameters(mapping, inplace=True)

        # Add inverse gate to QuantumCircuit
        qc.compose(c2q.inverse(), qc.qubits[-num_qubits:], inplace=True)

        return qc

    def _define(self):
        qregs = [QuantumRegister(n, name=f"dim{i}") for i, n in enumerate(self.dims_q)]
        qregs += [
            QuantumRegister(n, name=f"fltr{i}")
            for i, n in enumerate(self.filter_shape_q)
        ]
        qc = QuantumCircuit(*qregs, name=self.name)

        Convolution.shift(qc, self.filter_shape_q)

        fltr = np.array(self.params).reshape(self.filter_shape, order="F")
        # Convolution.filter(qc, fltr)

        Convolution.permute(qc, self.filter_shape_q)
        Convolution.filter_post_permute(qc, fltr)

        self.definition = qc
