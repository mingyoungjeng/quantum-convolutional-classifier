from unitary.unitary import Unitary
import pennylane as qml
from itertools import tee


class SimpleConvolution(Unitary):
    @staticmethod
    def _u(params, wires) -> None:
        params1, params2 = params.reshape((2, 3))

        # First Rot layer
        for wire in wires:
            qml.Rot(*params1, wires=wire)

        # CNOT gates
        control, target = tee(wires)
        first = next(target, None)
        for cnot_wires in zip(control, target):
            qml.CNOT(wires=cnot_wires)

        # Final CNOT gate (last to first)
        if len(wires) > 1:
            qml.CNOT(wires=(wires[-1], first))

        # Second Rot layer
        for wire in wires:
            qml.Rot(*params2, wires=wire)

    def total_params(self, *_) -> int:
        return 6


class SimplePooling(SimpleConvolution):
    @staticmethod
    def _u(params, wires) -> None:
        wires, target = SimplePooling.cut(-1)
        qml.cond(qml.measure(target) == 0, SimpleConvolution._u)(params, wires)

    def total_params(self, *_) -> int:
        return 6
