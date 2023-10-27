from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import tee

import pennylane as qml
from pennylane.operation import Operation
from pennylane.wires import Wires

from qcc.quantum.pennylane import Unitary

if TYPE_CHECKING:
    pass


def define_filter(op: type[Operation] = qml.RY, num_layers: int = 1):
    class Wrapper(FilterQML):
        def __init__(self, *params, wires: Wires, id=None, **_):
            super().__init__(*params, wires=wires, op=op, num_layers=num_layers, id=id)

        @staticmethod
        def _shape(num_wires: int, **_):
            return FilterQML._shape(num_wires, op=op, num_layers=num_layers)

    return Wrapper


class FilterQML(Unitary):
    """Alternative to C2Q for implementing convolutional filters in QML"""

    def __init__(
        self,
        *params,
        wires: Wires,
        op: type[Operation] = qml.RY,
        num_layers: int = 1,
        id=None,
        **_,
    ):
        self._hyperparameters = {"op": op, "num_layers": num_layers}

        super().__init__(*params, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(*params, wires, **hyperparameters):
        (params,) = params
        op: type[Operation] = hyperparameters.get("op", qml.RY)
        num_layers: int = hyperparameters.get("num_layers", 1)

        op_list = []
        shape = (num_layers, len(wires), op.num_params)
        params = params.reshape(shape)

        for i, angles in enumerate(params):
            # Rotation gate layer
            op_list += [op(*a, wire) for a, wire in zip(angles, wires)]

            if i >= len(params) - 1:
                continue

            # CNOT gates
            # wires = wires[::-1]
            control, target = tee(wires)
            first = next(target, None)
            op_list += [qml.CNOT(wires=w) for w in zip(control, target)]

            # Final CNOT gate (last to first)
            if len(wires) > 1:
                op_list += [qml.CNOT(wires=(wires[-1], first))]

        return op_list

    @staticmethod
    def _shape(num_wires: Wires, **hyperparameters) -> int:
        op: type[Operation] = hyperparameters.get("op", qml.RY)
        num_layers: int = hyperparameters.get("num_layers", 1)

        return num_layers * num_wires * op.num_params
