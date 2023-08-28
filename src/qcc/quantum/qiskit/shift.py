from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union

from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from numpy import sign

if TYPE_CHECKING:
    from typing import Optional


def _shift(qc: QuantumCircuit, k: int = 1, num_ctrl_qubits: int = 0) -> None:
    if k == 0:
        return

    controls, targets = qc.qubits[:num_ctrl_qubits], qc.qubits[num_ctrl_qubits:]

    for _ in range(abs(k)):
        for i in range(len(targets))[:: -sign(k)]:
            ctrls_i = controls + list(targets[:i])

            if len(ctrls_i) == 0:
                qc.x(targets[i])
            else:
                qc.mcx(ctrls_i, targets[i])


class Shift(Gate):
    def __init__(
        self, num_qubits: int, k: int = 1, label: Optional[str] = None
    ) -> None:
        super().__init__("shift", num_qubits, [k], label=label)

    @property
    def k(self):
        return self.params[0]

    def _define(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        _shift(qc, self.k)
        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: Optional[str] = None,
        ctrl_state: int | str | None = None,
    ):
        gate = CShift(
            self.num_qubits + num_ctrl_qubits,
            k=self.k,
            label=label,
            num_ctrl_qubits=num_ctrl_qubits,
            ctrl_state=ctrl_state,
        )
        gate.base_gate.label = self.label
        return gate

    def inverse(self):
        return self.__class__(self.num_qubits, k=-self.k)


class CShift(ControlledGate):
    def __init__(
        self,
        num_qubits: int,
        k: int = 1,
        label: str | None = None,
        num_ctrl_qubits: int = 1,
        ctrl_state: Optional[int | str] = None,
    ):
        base_gate = Shift(num_qubits - num_ctrl_qubits, k=k)
        super().__init__(
            "controlled shift",
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


class Decrementor(Shift):
    def __init__(self, num_qubits: int, k: int = 1, label: str | None = None) -> None:
        if k < 0:
            raise ValueError(f"k must be > 0, got {k}")

        super().__init__(num_qubits, -k, label)


class Incrementor(Shift):
    def __init__(self, num_qubits: int, k: int = 1, label: str | None = None) -> None:
        if k < 0:
            raise ValueError(f"k must be > 0, got {k}")

        super().__init__(num_qubits, k, label)
