from __future__ import annotations
from typing import TYPE_CHECKING

import pennylane as qml
from pennylane.operation import Operation, AnyWires

if TYPE_CHECKING:
    from typing import Iterable
    from qcc.quantum.pennylane import Wires


class Shift(Operation):
    num_wires = AnyWires

    @property
    def num_params(self) -> int:
        return 1

    @property
    def ndim_params(self) -> tuple[int]:
        return (0,)

    # TODO: optimize
    @staticmethod
    def compute_decomposition(*k: int, wires: Wires, **_) -> Iterable[Operation]:
        (k,) = k  # Keep the type-checker happy

        # k = list(f"{k:0{len(wires)}b}")

        op_list = []
        for i, w in enumerate(wires) if k < 0 else reversed(list(enumerate(wires))):
            if i == 0:
                op_list += [qml.PauliX(w)]
            else:
                op_list += [qml.MultiControlledX(wires=list(wires[:i]) + [w])]

        return op_list

    def adjoint(self) -> Operation:
        return Shift(-self.parameters, self.qubits.flatten())
