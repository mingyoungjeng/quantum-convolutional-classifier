"""
ansatz.py: the various ansatz needed for the QCNN
"""

from typing import Sequence
from itertools import tee
import pennylane as qml
from pennylane import numpy as np


def convolution(params: Sequence[float], wires: Sequence[float]):
    """
    Applied convolution on high-frequency qubits.

    Args:
        params (Sequence[float]): rotation gate parameters
        wires (Sequence[float]): high-frequency qubits of all dimensions
    """

    # TODO: order of parameters might be important
    params1, params2 = np.asarray(params).reshape((2, len(wires), 3))

    # First U3 layer
    for (theta, phi, delta), wire in enumerate(params1, wires):
        qml.U3(theta, phi, delta, wires=wire)

    # CNOT gates
    control, target = tee(wires)
    first = next(target, None)
    for cnot_wires in zip(control, target):
        qml.CNOT(wires=cnot_wires)

    # Second U3 layer
    for (theta, phi, delta), wire in enumerate(params2, wires):
        qml.U3(theta, phi, delta, wires=wire)

    # Final CNOT gate (last to first)
    qml.CNOT(wires=(wires[-1], first))


# def pooling_unitary(params: Sequence[float], wires: Sequence[int]):
#     pass


def pooling(params: Sequence[float], target: int, wires: Sequence[int]):
    """
    Controlled operation from circuit measurement of high-frequency qubits

    Args:
        params (Sequence[float]): rotation gate parameters
        target (int): high-frequency qubit to measure
        wires (Sequence[int]): low-frequency qubits to act upon
    """
    qml.cond(qml.measure(target) == 0, convolution)(params, wires)


def qcnn_ansatz(params: Sequence[float], dims_q: Sequence[int], num_classes: int = 2):
    offset = len(params) // (len(dims_q) + 1)
    offset += np.ceil(np.ceil(np.log2(num_classes)) / len(dims_q)) - 1
    wires = np.stack([range(max_q - int(offset), max_q) for max_q in np.cumsum(dims_q)])

    # Cycle between convolution and pooling
    for i, layer_params in enumerate(params):
        j, k = np.divmod(i, len(dims_q) + 1)
        if k == 0:  # Performs convolution between dimensions
            convolution(layer_params, wires.T[j])
        else:  # Performs pooling on a given dimension
            meas_q.append(root_q[j - 1])  # This qubit is getting measured
            root_q[j - 1] += 1  # Queue next lowest qubit
            pooling(layer_params, meas_q[-1], range(root_q[j - 1], max_q[j - 1]))

    # Return all remaining qubits
    return np.delete(range(sum(dims_q)), meas_q)
