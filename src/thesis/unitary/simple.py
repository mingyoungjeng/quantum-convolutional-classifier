from unitary.unitary import Unitary
import pennylane as qml
from itertools import tee
import numpy as np


class SimpleConvolution(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        op_list = []
        params1, params2 = params.reshape((2, 3))

        # First Rot layer
        op_list += [qml.Rot(*params1, wires=wire) for wire in wires]

        # CNOT gates
        control, target = tee(wires)
        first = next(target, None)
        op_list += [qml.CNOT(wires=cnot_wires) for cnot_wires in zip(control, target)]

        # Final CNOT gate (last to first)
        if len(wires) > 1:
            op_list += [qml.CNOT(wires=(wires[-1], first))]

        # Second Rot layer
        op_list += [qml.Rot(*params2, wires=wire) for wire in wires]

    @staticmethod
    def shape(*_) -> int:
        return 6


class SimplePooling(SimpleConvolution):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        wires, target, *_ = np.split(wires, [-1])
        return [qml.cond(qml.measure(target) == 0, SimpleConvolution)(params, wires)]

    @staticmethod
    def shape(*_) -> int:
        return 6
