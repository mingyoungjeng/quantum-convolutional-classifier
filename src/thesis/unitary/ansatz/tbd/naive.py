from typing import Sequence
from numbers import Number
from itertools import tee
import numpy as np
import pennylane as qml

# from pennylane import numpy as np

from ansatz.convolution import Ansatz


class NaiveAnsatz(Ansatz):
    conv_params = 3
    pool_params = 3

    @staticmethod
    def convolution(params: Sequence[Number], wires: Sequence[Number]):
        """
        Applied convolution on high-frequency qubits.

        Args:
            params (Sequence[Number]): rotation gate parameters
            wires (Sequence[Number]): high-frequency qubits of all dimensions
        """

        # TODO: order of parameters might be important
        params = params.reshape((len(wires), 3))

        # First Rot layer
        for (theta, phi, delta), wire in zip(params, wires):
            qml.Rot(theta, phi, delta, wires=wire)

        # CNOT gates
        control, target = tee(wires)
        first = next(target, None)
        for cnot_wires in zip(control, target):
            qml.CNOT(wires=cnot_wires)

        # Final CNOT gate (last to first)
        if len(wires) > 1:
            qml.CNOT(wires=(wires[-1], first))

    # def pooling_unitary(params: Sequence[float], wires: Sequence[int]):
    #     pass

    @staticmethod
    def pooling(params: Sequence[Number], target: int, wires: Sequence[int]):
        """
        Controlled operation from circuit measurement of high-frequency qubits

        Args:
            params (Sequence[float]): rotation gate parameters
            target (int): high-frequency qubit to measure
            wires (Sequence[int]): low-frequency qubits to act upon
        """
        qml.cond(qml.measure(target) == 0, qml.Rot)(*params, wires)

    def __call__(
        self,
        params: Sequence[Number],
        dims_q: Sequence[int],
        num_layers: int = 1,
        num_classes: int = 2,
    ) -> Sequence[int]:
        max_wires = np.cumsum(dims_q)
        wires = range(sum(dims_q))

        for i in range(num_layers):
            # Apply convolution layers
            for j in 1 + np.arange(min(dims_q)):
                conv_params, params = np.split(params, [3 * len(dims_q)])
                self.convolution(conv_params, max_wires - j)

        conv_params, params = np.split(params, [3 * len(wires)])
        self.convolution(conv_params, wires)

        params, _ = np.split(params, [3 * len(wires)])
        params = params.reshape((len(wires), 3))
        for (theta, phi, delta), wire in zip(params, wires):
            qml.Rot(theta, phi, delta, wires=wire)

        return wires

    def total_params(
        self, dims_q: Sequence[int], num_layers: int = 1, num_classes: int = 2
    ):
        n_params = self.conv_params * len(dims_q) * min(dims_q) * num_layers
        n_params += self.conv_params * sum(dims_q) * 2

        return n_params
