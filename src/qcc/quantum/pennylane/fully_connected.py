from __future__ import annotations
from typing import TYPE_CHECKING

import pennylane as qml

from qcc.quantum import to_qubits
from qcc.quantum.pennylane import Unitary

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

            for j, (ctrl, wire) in enumerate(zip(controls, wires)):
                params_j = params_i[2 * j : 2 * (j + 1)]
                ops = tuple(qml.RY(theta, wires=wire) for theta in params_j)
                op_list += [qml.Select(ops, ctrl)]

        return op_list

    @staticmethod
    def _shape(num_wires: Wires, **_) -> int:
        total = 0
        for _ in range(to_qubits(num_wires)):
            total += num_wires // 2
            num_wires = (num_wires + 1) // 2
        return 2 * total

        # return 2 * to_qubits(len(wires))


class FullyConnectedSimple(Unitary):
    @staticmethod
    def compute_decomposition(*params, wires, **_):
        # Keep the type-checker happy
        (params,) = params

        op_list = []
        num_layers = to_qubits(len(wires))
        for i in range(num_layers):
            wires, controls = wires[0::2], wires[1::2]

            params_i = params[2 * i : 2 * (i + 1)]
            for ctrl, wire in zip(controls, wires):
                ops = tuple(qml.RY(theta, wires=wire) for theta in params_i)
                op_list += [qml.Select(ops, ctrl)]

        return op_list

    @staticmethod
    def _shape(num_wires: Wires, **_) -> int:
        return 2 * to_qubits(num_wires)
