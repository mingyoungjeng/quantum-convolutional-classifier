from __future__ import annotations
from typing import TYPE_CHECKING

import pennylane as qml
from pennylane.operation import Operation, AnyWires

if TYPE_CHECKING:
    from typing import Sequence
    from thesis.unitary import Wires


class Shift(Operation):
    num_wires = AnyWires

    @property
    def num_params(self) -> int:
        return 1

    @property
    def ndim_params(self) -> tuple[int]:
        return (0,)

    @staticmethod
    def compute_decomposition(*k: int, wires: Wires, **_) -> Sequence[Operation]:
        op_list = []
        for _ in range(abs(k)):
            for i, w in enumerate(wires) if k < 0 else reversed(list(enumerate(wires))):
                if i == 0:
                    op_list += [qml.PauliX(w)]
                else:
                    op_list += [qml.MultiControlledX(wires=list(wires[:i]) + [w])]

        return op_list

    def adjoint(self) -> Operation:
        return Shift(-self.parameters, self.wires)
