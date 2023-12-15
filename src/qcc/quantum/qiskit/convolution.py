"""
Multidimensional Convolution Operation

TODO: Replace StatePreparation with MAC/C2Q
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from numbers import Number
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Gate
from qiskit.circuit.library import StatePreparation

from qcc.quantum import to_qubits
from qcc.quantum.qiskit.shift import Shift

if TYPE_CHECKING:
    from typing import Sequence


class Convolution(Gate):
    """Multidimensional Convolution Operation"""

    __slots__ = "dims", "kernel_dims", "kernel_type"

    dims: Sequence[int]
    kernel_dims: Sequence[int]
    kernel_type: type[np.ndarray]

    def __init__(
        self,
        dims: int | Sequence[int],
        kernel: np.ndarray,
        label: str | None = None,
    ) -> None:
        self.dims = [dims] if isinstance(dims, Number) else dims
        self.kernel_dims = kernel.shape
        self.kernel_type = type(kernel)

        num_data_qubits = sum(self._dims_q)
        num_kernel_qubits = sum(self._kernel_dims_q)
        num_qubits = int(num_data_qubits + num_kernel_qubits)
        super().__init__("convolution", num_qubits, kernel.T.ravel(), label)

    @property
    def dim(self):
        return len(self.kernel_dims)

    @property
    def _dims_q(self):
        return to_qubits(self.dims)

    @property
    def _kernel_dims_q(self):
        return to_qubits(self.kernel_dims)

    @staticmethod
    def stride(qc: QuantumCircuit, kernel_dims_q, stride=1, H=True) -> QuantumCircuit:
        for i, kdq in enumerate(kernel_dims_q):
            data_qubits = qc.qregs[i]
            kernel_qubits = qc.qregs[i - len(kernel_dims_q)][:kdq]

            if len(data_qubits) == 0:
                continue

            # ==== apply H gate(s) to kernel qubits ==== #
            if H:
                qc.h(kernel_qubits)

            # ==== shift operation ==== #
            for j, control in enumerate(kernel_qubits):
                gate = Shift(k=-stride, num_qubits=len(data_qubits[j:])).control()
                qc.compose(gate, [control, *data_qubits[j:]], inplace=True)

        return qc

    @staticmethod
    def permute(qc: QuantumCircuit, kernel_dims_q):
        for i, kdq in enumerate(kernel_dims_q):
            data_qubits = qc.qregs[i][:kdq]
            kernel_qubits = qc.qregs[i - len(kernel_dims_q)][:kdq]

            for d, k in zip(data_qubits, kernel_qubits):
                qc.swap(d, k)

        return qc

    # @staticmethod
    # def mac(qc: QuantumCircuit, kernel: np.ndarray) -> QuantumCircuit:

    #     mac = StatePreparation(kernel.conj(), inverse=True, normalize=True)
    #     qc.compose(mac, inplace=True)

    #     return qc

    def _define(self) -> None:
        qregs = [QuantumRegister(n, name=f"dim{i}") for i, n in enumerate(self._dims_q)]
        qregs += [
            QuantumRegister(n, name=f"kernel{i}")
            for i, n in enumerate(self._kernel_dims_q)
        ]
        qc = QuantumCircuit(*qregs, name=self.name)

        Convolution.stride(qc, self._kernel_dims_q)

        Convolution.permute(qc, self._kernel_dims_q)

        # ==== MAC ==== #
        kernel_qubits = qc.qregs[-len(self.kernel_dims) :]
        kernel_qubits = [qubit for qreg in kernel_qubits for qubit in qreg]
        MAC = StatePreparation(np.conj(self.params), inverse=True, normalize=True)
        qc.compose(MAC, qubits=kernel_qubits, inplace=True)

        self.definition = qc

    def inverse(self) -> Gate:
        # Define an inverse of your gate if you are a nice person
        pass
