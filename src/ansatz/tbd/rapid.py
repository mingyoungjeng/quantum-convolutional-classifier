from typing import Sequence
from numbers import Number
from itertools import tee
import numpy as np
import pennylane as qml

# from pennylane import numpy as np

from ansatz.convolution import Ansatz


class RapidAnsatz(Ansatz):
    conv_params = 6
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
        n_dim = min(dims_q)  # Min number of qubits per dimension

        for i in range(num_layers):
            # Apply convolution operations on all wires
            conv_params, params = np.split(params, [self.conv_params])
            for j in 1 + np.arange(n_dim):
                # conv_params, params = np.split(params, [self.conv_params]) # * len(dims_q)
                self.convolution(conv_params, max_wires - j)

            # Pooling
            half, mod = np.divmod(n_dim, 2)
            for max_wire in max_wires:
                all_wires = range(max_wire - n_dim, max_wire)

                measurements = all_wires[:half][::-1]
                controlled = all_wires[half:][::-1]

                pool_params, params = np.split(params, [self.pool_params])
                for m_wire, t_wire in zip(measurements, controlled):
                    # pool_params, params = np.split(params, [self.pool_params])
                    self.pooling(pool_params, m_wire, t_wire)
            n_dim = half + mod

        # Fully connected layer
        meas = np.array(
            [np.arange(max_wire - n_dim, max_wire) for max_wire in max_wires]
        ).flatten(order="F")

        conv_params, params = np.split(params, [self.conv_params])  # * len(meas)
        self.convolution(conv_params, meas)

        return meas

    def total_params(
        self, dims_q: Sequence[int], num_layers: int = 1, num_classes: int = 2
    ):
        n_params = 0
        n_dim = min(dims_q)  # Min number of qubits per dimension

        for i in range(num_layers):
            n_params += self.conv_params  # * n_dim  # * len(dims_q)
            half, mod = np.divmod(n_dim, 2)
            n_params += self.pool_params * len(dims_q)  # * half
            n_dim = half + mod

        n_params += self.conv_params  # * n_dim * len(dims_q)

        return n_params
