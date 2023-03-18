"""
ansatz.py: the various ansatz needed for the QCNN
"""

from typing import Sequence
from numbers import Number
import numpy as np
from itertools import tee
import pennylane as qml

# from pennylane import numpy as np


def convolution(params: Sequence[Number], wires: Sequence[Number]):
    """
    Applied convolution on high-frequency qubits.

    Args:
        params (Sequence[Number]): rotation gate parameters
        wires (Sequence[Number]): high-frequency qubits of all dimensions
    """

    # TODO: order of parameters might be important
    params1, params2 = params.reshape((2, 3))

    # First U3 layer
    for wire in wires:
        qml.U3(*params1, wires=wire)

    # CNOT gates
    control, target = tee(wires)
    first = next(target, None)
    for cnot_wires in zip(control, target):
        qml.CNOT(wires=cnot_wires)

    # Second U3 layer
    for wire in wires:
        qml.U3(*params2, wires=wire)

    # Final CNOT gate (last to first)
    if len(wires) > 1:
        qml.CNOT(wires=(wires[-1], first))


# def pooling_unitary(params: Sequence[float], wires: Sequence[int]):
#     pass


def pooling(params: Sequence[Number], target: int, wires: Sequence[int]):
    """
    Controlled operation from circuit measurement of high-frequency qubits

    Args:
        params (Sequence[float]): rotation gate parameters
        target (int): high-frequency qubit to measure
        wires (Sequence[int]): low-frequency qubits to act upon
    """
    qml.cond(qml.measure(target) == 0, convolution)(params, wires)


def qcnn_ansatz(
    params: Sequence[Number],
    dims_q: Sequence[int],
    num_layers: int = 1,
    num_classes: int = 2,
) -> Sequence[int]:
    max_wires = np.cumsum(dims_q)
    offset = -int(np.log2(num_classes) // -len(dims_q))  # Ceiling division
    wires = max_wires - offset

    for i in range(num_layers):
        # Apply convolution layers
        for j in 1 + np.arange(min(dims_q)):
            conv_params, params = np.split(params, [6])
            convolution(conv_params, max_wires - j)

    # Qubits to measure
    meas = np.array(
        [
            np.arange(target_wire, max_wire)
            for target_wire, max_wire in zip(wires, max_wires)
        ]
    ).flatten(order="F")

    # Return the minimum required number of qubits to measure in order
    return np.sort(meas[: int(np.ceil(np.log2(num_classes)))])
