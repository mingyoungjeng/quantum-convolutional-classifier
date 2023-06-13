from __future__ import annotations
from typing import TYPE_CHECKING

import pennylane as qml

from qcnn.quantum import to_qubits
from qcnn.quantum.operation import Unitary, Multiplex

if TYPE_CHECKING:
    from pennylane.wires import Wires


class FullyConnected(Unitary):
    @staticmethod
    def compute_decomposition(*params, wires, **_):
        # Keep the type-checker happy
        (params,) = params

        op_list = []
        num_layers = to_qubits(len(wires))
        for i in range(num_layers):
            wires, controls = wires[0::2], wires[1::2]
            params_i, params = params[: 2 * len(controls)], params[2 * len(controls) :]

            op_list += [
                Multiplex(params_i[2 * j : 2 * (j + 1)], wire, ctrl, qml.RY)
                for j, (ctrl, wire) in enumerate(zip(controls, wires))
            ]

        return op_list

    @staticmethod
    def _shape(wires: Wires) -> int:
        n = len(wires)
        total = 0
        for _ in range(to_qubits(len(wires))):
            total += n // 2
            n = (n + 1) // 2
        return 2 * total

        # return 2 * to_qubits(len(wires))
