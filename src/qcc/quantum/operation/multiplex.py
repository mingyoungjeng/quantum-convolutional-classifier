from __future__ import annotations
from typing import TYPE_CHECKING, Iterator

from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires
from pennylane.ops import Controlled

from qcc.quantum import binary

if TYPE_CHECKING:
    from typing import Iterable, Optional, Mapping


class Multiplex(Operation):
    num_wires = AnyWires

    def __init__(
        self,
        params,
        wires: Wires,
        control_wires: Optional[Wires] = None,
        op: type[Operation] = Operation,
        hyperparameters: Optional[Mapping] = None,
        do_queue=True,
        id=None,
    ):
        wires = Wires(wires)
        control_wires = [] if control_wires is None else Wires(control_wires)

        self._hyperparameters = {
            "control_wires": control_wires,
            "op": op,
            "hyperparameters": {} if hyperparameters is None else hyperparameters,
        }

        if len(control_wires) > 0:
            wires = control_wires + wires

        super().__init__(*params, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(
        *params: Iterable, wires: Wires, **hyperparameters
    ) -> Iterable[Operation]:
        # Keep the type-checker happy
        ctrls: Optional[Wires] = hyperparameters["control_wires"]
        op: type[Operation] = hyperparameters["op"]
        hyperparameters: Mapping = hyperparameters["hyperparameters"]

        if len(ctrls) == 0:
            return [op(*params, wires=wires, **hyperparameters)]

        # TODO: cleanup
        wires = wires[len(ctrls) :]
        return [
            Controlled(
                op(
                    *param,
                    wires=wires,
                    **hyperparameters,
                )
                if isinstance(param, (tuple, Iterator))
                else op(
                    param,
                    wires=wires,
                    **hyperparameters,
                ),
                ctrls,
                binary(i, len(ctrls))[::-1],
            )
            for i, param in enumerate(params)
        ]

    def adjoint(self) -> Operation:
        return Multiplex(
            self.hyperparameters["op"], -self.parameters, self.qubits.flatten()
        )
