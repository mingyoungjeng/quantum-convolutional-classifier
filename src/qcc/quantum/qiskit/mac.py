"""
Multiply-and-Accumulate

TODO: Using complex kernels is something I've generally left unexplored. Not sure if to conj the row vector or not.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit

from qcc.quantum import to_qubits

if TYPE_CHECKING:
    from torch import Tensor


def mac(qc: QuantumCircuit, psi_in: np.ndarray | Tensor) -> None:
    """Multiply-and-Accumulate Operation"""

    params = get_params(psi_in)

    phi = next(params)
    theta = next(params)
    t = next(params)

    for j, target in enumerate(qc.qubits):
        control = qc.qubits[j + 1 :]

        if j == 0 and phi.any():
            qc.ucrz(list(-phi), control, target)

        if j > 0:
            theta = next(params)
        qc.ucry(list(-theta), control, target)

        if j == 0 and t.any():
            qc.ucrz(list(t), control, target)


def get_params(x_in: np.ndarray | Tensor):
    """Returns C2Q parameters in reverse order"""

    p = x_in
    num_qubits = to_qubits(len(x_in))
    for i in range(num_qubits):
        x = p.reshape((len(p) // 2, 2))
        p = x.norm(dim=1) if hasattr(x, "norm") else np.linalg.norm(x, axis=1)
        x = (x / (p[:, None] + 1e-12)).T  # TODO: do the [1, 0] replacement

        # ==== Rz angles ==== #
        if i == 0:
            phase = x.angle() if hasattr(x, "angle") else np.angle(x)
            phi = phase[1] - phase[0]
            t = phase[1] + phase[0]

            yield phi

        # ==== Ry angles ==== #
        theta = abs(x)
        theta = theta[0] + 1j * theta[1]
        theta = theta.angle() if hasattr(theta, "angle") else np.angle(theta)
        theta = 2 * theta
        yield theta

        # ==== Rz angles ==== #
        if i == 0:
            yield t


class MultiplyAndAccumulate(Gate):
    """MAC Operation"""

    def __init__(
        self,
        params: np.ndarray | Tensor,
        label: str | None = "$U_\\text{MAC}$",
    ) -> None:
        num_qubits = int(to_qubits(len(params)))
        super().__init__("MAC", num_qubits, params, label)

    def _define(self) -> None:
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        mac(qc, np.array(self.params))

        self.definition = qc

    # def inverse(self) -> Gate:
    #     # Define an inverse of your gate if you are a nice person
    #     pass
