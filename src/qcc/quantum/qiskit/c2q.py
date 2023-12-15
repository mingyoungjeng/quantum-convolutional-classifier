"""Classical-to-Quantum (C2Q) Arbitrary State Synthesis"""

from __future__ import annotations
from typing import TYPE_CHECKING

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit

from qcc.quantum import to_qubits
from qcc.quantum.qiskit.mac import MultiplyAndAccumulate

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor


class C2Q(Gate):
    """C2Q Operation"""

    def __init__(
        self,
        params: np.ndarray | Tensor,
        label: str | None = None,
    ) -> None:
        num_qubits = int(to_qubits(len(params)))
        super().__init__("C2Q", num_qubits, params, label)

    def _define(self) -> None:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.compose(self.inverse().inverse(), inplace=True)
        self.definition = qc

    def inverse(self):
        return MultiplyAndAccumulate(self.params)
