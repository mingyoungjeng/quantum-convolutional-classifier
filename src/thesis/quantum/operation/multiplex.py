from __future__ import annotations
from typing import TYPE_CHECKING

import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires

from thesis.quantum import binary

if TYPE_CHECKING:
    from typing import Iterable


class Multiplex(Operation):
    num_wires = AnyWires

    def __init__(
        self,
        params,
        wires: Wires,
        op: type[Operation] = Operation,
        do_queue=True,
        id=None,
    ):
        self._hyperparameters = {"op": op}

        super().__init__(params, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(
        *params: Iterable, wires: Wires, op: type[Operation]
    ) -> Iterable[Operation]:
        (ps,) = params  # Keep the type-checker happy

        c, w = wires[: -op.num_wires], wires[-op.num_wires :]

        if len(c) == 0:
            return op(ps, wires)
        else:
            return [qml.ctrl(op, c, binary(i, len(c)))(p, w) for i, p in enumerate(ps)]

    def adjoint(self) -> Operation:
        return Multiplex(self.hyperparameters["op"], -self.parameters, self.wires)
