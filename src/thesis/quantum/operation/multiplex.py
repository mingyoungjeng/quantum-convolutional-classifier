from __future__ import annotations
from typing import TYPE_CHECKING

import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires
from pennylane.ops import Controlled

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
        *params: Iterable, wires: Wires, **hyperparameters
    ) -> Iterable[Operation]:
        # Keep the type-checker happy
        (params,) = params
        op: type[Operation] = hyperparameters["op"]

        ctrls, wires = wires[: -op.num_wires], wires[-op.num_wires :]

        if len(ctrls) == 0:
            return [op(params[0], wires)]
        else:
            return [
                Controlled(op(param, wires), ctrls, binary(i, len(ctrls))[::-1])
                for i, param in enumerate(params)
            ]

    def adjoint(self) -> Operation:
        return Multiplex(self.hyperparameters["op"], -self.parameters, self.wires)
