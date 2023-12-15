"""_summary_"""

from __future__ import annotations
from typing import TYPE_CHECKING

from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from numpy import sign

from qcc.quantum import to_qubits

if TYPE_CHECKING:
    pass


def _shift(qc: QuantumCircuit, k: int = 1, num_ctrl_qubits: int = 0) -> None:
    """
    _summary_

    Args:
        qc (QuantumCircuit): _description_
        k (int, optional): _description_. Defaults to 1.
        num_ctrl_qubits (int, optional): _description_. Defaults to 0.
    """

    if k == 0:
        return

    k, sgn = abs(k), sign(k)
    controls, targets = qc.qubits[:num_ctrl_qubits], qc.qubits[num_ctrl_qubits:]

    # ==== calculate the number of bits to use in binary decomposition of shift ==== #
    # k+1 so powers of 2 are represented with the correct number of bits
    #   Ex: 2 = 011, and ceil(log2(3)) = 2
    #       4 = 100, but ceil(log2(4)) = 2
    num_bits = min(to_qubits(k + 1), len(targets))

    # ==== perform shift operation by ±k ==== #
    # Big or little endian depending on which minimizes depth [::sgn]
    for i in range(num_bits)[::sgn]:
        k_i = k // 2**i % 2
        if k_i == 0:
            continue

        # ==== shift by ±1, see [::-sgn] ==== #
        for j, target in tuple(enumerate(targets[i:]))[::-sgn]:
            ctrls_j = controls + targets[i : i + j]

            if len(ctrls_j) == 0:
                qc.x(target)
            else:
                qc.mcx(ctrls_j, target)


class Shift(Gate):
    """_summary_"""

    def __init__(
        self,
        num_qubits: int,
        k: int = 1,
        label: str | None = None,
    ) -> None:
        if label is None:
            label = f"$U_\\text{{Shift}}$"  # ^{{{k}}}
        super().__init__("shift", num_qubits, [k], label=label)

    @property
    def k(self):
        return self.params[0]

    def _define(self) -> None:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        _shift(qc, self.k)
        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
    ) -> ControlledGate:
        gate = CShift(
            self.num_qubits + num_ctrl_qubits,
            k=self.k,
            label=label,
            num_ctrl_qubits=num_ctrl_qubits,
            ctrl_state=ctrl_state,
        )
        gate.base_gate.label = self.label
        return gate

    def inverse(self) -> Shift:
        return self.__class__(self.num_qubits, k=-self.k)


class CShift(ControlledGate):
    """_summary_"""

    def __init__(
        self,
        num_qubits: int,
        k: int = 1,
        label: str | None = None,
        num_ctrl_qubits: int = 1,
        ctrl_state: int | str | None = None,
    ):
        base_gate = Shift(num_qubits - num_ctrl_qubits, k=k)
        super().__init__(
            "shift",
            num_qubits,
            [k],
            label=label,
            num_ctrl_qubits=num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=base_gate,
        )

    @property
    def k(self):
        return self.params[0]

    def _define(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        _shift(qc, self.k, num_ctrl_qubits=self.num_ctrl_qubits)
        self.definition = qc

    def inverse(self):
        return self.__class__(
            self.num_qubits,
            k=-self.k,
            num_ctrl_qubits=self.num_ctrl_qubits,
            ctrl_state=self.ctrl_state,
        )


class Incrementor(Shift):
    """_summary_"""

    def __init__(self, num_qubits: int, k: int = 1, label: str | None = None) -> None:
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")

        super().__init__(num_qubits, k, label)
        self.name = "incrementer"


class Decrementor(Shift):
    """_summary_"""

    def __init__(self, num_qubits: int, k: int = 1, label: str | None = None) -> None:
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")

        super().__init__(num_qubits, -k, label)
        self.name = "decrementer"
