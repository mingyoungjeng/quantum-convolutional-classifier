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


def qcnn_ansatz(params, dims):
    last_params = params.pop()

    for i, (conv_params, pool_params) in enumerate(params):
        # Apply convolution over all high-frequency qubits
        convolution(conv_params, dims[i])

        # for loop
        for dims_j in dims.T:
            pooling(pool_params, dims_j[-i], dims_j[: 1 - i])

    # Final layer of convolution
    convolution(last_params)
