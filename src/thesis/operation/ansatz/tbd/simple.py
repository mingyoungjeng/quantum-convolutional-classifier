from typing import Iterable
from numbers import Number
import numpy as np
from itertools import tee
import pennylane as qml
from ansatz.convolution import Ansatz

# from pennylane import numpy as np


class SimpleAnsatz(Ansatz):
    conv_params = 6
    pool_params = 6

    @staticmethod
    def convolution(params: Iterable[Number], wires: Iterable[Number]):
        """
        Applied convolution on high-frequency qubits.

        Args:
            params (Iterable[Number]): rotation gate parameters
            wires (Iterable[Number]): high-frequency qubits of all dimensions
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

    # def pooling_unitary(params: Iterable[float], wires: Iterable[int]):
    #     pass

    @staticmethod
    def pooling(params: Iterable[Number], target: int, wires: Iterable[int]):
        """
        Controlled operation from circuit measurement of high-frequency qubits

        Args:
            params (Iterable[float]): rotation gate parameters
            target (int): high-frequency qubit to measure
            wires (Iterable[int]): low-frequency qubits to act upon
        """
        qml.cond(qml.measure(target) == 0, SimpleAnsatz.convolution)(params, wires)

    def __call__(
        self,
        params: Iterable[Number],
        dims_q: Iterable[int],
        num_layers: int = 1,
        num_classes: int = 2,
    ) -> Iterable[int]:
        max_wires = np.cumsum(dims_q)
        offset = -int(np.log2(num_classes) // -len(dims_q))  # Ceiling division
        wires = max_wires - offset

        for i in reversed(
            range(1, num_layers)
        ):  # Final behavior has different behavior
            # Apply convolution layer:
            conv_params, params = np.split(params, [6])
            pool_params, params = np.split(params, [6])
            # print(f"layer {i}: {len(conv_params)}-param convolution on wires {wires - i}")
            self.convolution(conv_params, wires - i)

            # Apply pooling layers
            for j, (target_wire, max_wire) in enumerate(zip(wires, max_wires)):
                # pool_params, params = np.split(params, [6 * ])
                # # print(
                # #     f"layer {i}: {len(pool_params)}-param pool {j} on wires {target_wire-i, target_wire-i+1}"
                # # )
                # pooling(pool_params, target_wire - i, target_wire - i + 1)
                # print(
                #     f"layer {i}: {len(pool_params)}-param pool {j} on wires {target_wire-i, range(target_wire-i+1, max_wire)}"
                # )
                self.pooling(
                    pool_params, target_wire - i, range(target_wire - i + 1, max_wire)
                )

        # Qubits to measure
        meas = np.array(
            [
                np.arange(target_wire, max_wire)
                for target_wire, max_wire in zip(wires, max_wires)
            ]
        ).flatten(order="F")

        # Final layer of convolution
        conv_params = params[: 6 * len(meas)]
        self.convolution(conv_params, np.sort(meas))

        # Return the minimum required number of qubits to measure in order
        # return np.sort(meas[: int(np.ceil(np.log2(num_classes)))])
        return np.sort(meas)

    def total_params(
        self, dims_q: Iterable[int], num_layers: int = 1, num_classes: int = 2
    ):
        n_conv_params = self.conv_params * num_layers
        n_pool_params = self.pool_params * (num_layers - 1)

        return n_conv_params + n_pool_params
