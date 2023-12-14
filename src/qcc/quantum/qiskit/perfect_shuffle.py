"""_summary_"""

from __future__ import annotations
from typing import TYPE_CHECKING

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit

if TYPE_CHECKING:
    pass


def _perfect_shuffle(
    qc: QuantumCircuit,
    start: int,
    end: int,
    direction: str = "right",
) -> None:
    """
    _summary_

    Args:
        qc (QuantumCircuit): _description_
        start (int): _description_
        end (int): _description_
        direction (str, optional): _description_. Defaults to "right".
    """

    rng = range(start, end)

    if direction == "left":
        rng = reversed(rng)

    for y in rng:
        qc.swap(y, y + 1)


class PerfectShuffle(Gate):
    """_summary_"""

    __slots__ = "direction"

    def __init__(
        self,
        num_qubits: int,
        direction: str = "right",
        label: str | None = None,
    ) -> None:
        self.direction = direction
        super().__init__("QPS", num_qubits, list(), label)

    def _define(self) -> None:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        _perfect_shuffle(qc, 0, self.num_qubits - 1, direction=self.direction)
        self.definition = qc

    def inverse(self) -> PerfectShuffle:
        direction = "right" if self.direction == "left" else "left"
        return self.__class__(self.num_qubits, direction=direction)


class RotateRight(PerfectShuffle):
    """_summary_"""

    def __init__(self, num_qubits: int, label: str | None = None) -> None:
        super().__init__(num_qubits, direction="right", label=label)
        self.name = "RoR"


class RotateLeft(PerfectShuffle):
    """_summary_"""

    def __init__(self, num_qubits: int, label: str | None = None) -> None:
        super().__init__(num_qubits, direction="left", label=label)
        self.name = "RoL"
