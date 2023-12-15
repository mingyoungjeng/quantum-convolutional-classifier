"""
Quantum Haar Transform as a quantum gate

TODO: Implement Pyramidal decomposition
TODO: Implement inverse QHT
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from numbers import Number
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit

from qcc.quantum import to_qubits
from qcc.quantum.qiskit.perfect_shuffle import RotateRight

if TYPE_CHECKING:
    from typing import Sequence


def _quantum_haar_transform(
    qc: QuantumCircuit,
    dims: Sequence[int],
    decomposition_levels: int | Sequence[int],
    method: str = "packet",
) -> None:
    dims_q = [to_qubits(x) for x in dims]
    if isinstance(decomposition_levels, Number):
        decomposition_levels = [decomposition_levels] * len(dims)

    # ==== wavelet decomposition using Hadamard gates ==== #
    base = np.hstack(([0], np.cumsum(dims_q[:-1])))
    for q, l in zip(base, decomposition_levels):
        qc.h(qc.qubits[q : q + l])

    # ==== data rearrangement ==== #

    # ==== pyramidal ==== #
    if method == "pyramidal":
        return NotImplementedError("TODO: implement pyramidal decomposition")

    # ==== packet ==== #
    top = np.cumsum(dims_q)
    for b, t, l in zip(base, top, decomposition_levels):
        RoR = RotateRight(int(t - b))
        for i in range(l):
            qc.append(RoR, qargs=qc.qubits[b:t])


class QuantumHaarTransform(Gate):
    """Quantum Haar Transform"""

    __slots__ = "dims", "decomposition_levels", "method"

    def __init__(
        self,
        dims: int | Sequence[int],
        decomposition_levels: int | Sequence[int] = 0,
        method: str = "packet",
        label: str | None = None,
    ) -> None:
        """
        _summary_

        Args:
            dims (int | Sequence[int]): _description_
            decomposition_levels (int | Sequence[int], optional): _description_. Defaults to 0.
            method (str, optional): _description_. Defaults to "packet".
            label (str | None, optional): _description_. Defaults to None.
        """

        self.dims = [dims] if isinstance(dims, Number) else dims
        self.decomposition_levels = decomposition_levels
        self.method = method

        num_qubits = sum(to_qubits(x) for x in self.dims)
        super().__init__("QHT", int(num_qubits), list(), label)

    def _define(self) -> None:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        _quantum_haar_transform(
            qc,
            dims=self.dims,
            decomposition_levels=self.decomposition_levels,
            method=self.method,
        )
        self.definition = qc

    def inverse(self) -> Gate:
        return NotImplementedError("TODO: write inverse QHT gate")
