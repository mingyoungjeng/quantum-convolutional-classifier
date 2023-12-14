"""
Use this file as a template for implementing new gates in Qiskit

See <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.Gate> for IBM's documentation
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit

if TYPE_CHECKING:
    pass


def _new_gate(qc: QuantumCircuit, *args, **kwargs) -> None:
    """Your gate as a function"""

    pass


class NewGate(Gate):
    """Your quantum gate"""

    def __init__(
        self,
        num_qubits: int,
        params: list,
        label: str | None = None,
    ) -> None:
        super().__init__("NewGate", num_qubits, params, label)

    def _define(self) -> None:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        # Call your gate definition function here
        _new_gate(qc)

        self.definition = qc

    def inverse(self) -> Gate:
        # Define an inverse of your gate if you are a nice person
        pass
