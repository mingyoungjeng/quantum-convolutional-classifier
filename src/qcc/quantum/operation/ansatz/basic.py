from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import tee
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from qcc.quantum.operation.ansatz import Ansatz
from qcc.quantum.operation import Unitary
from qcc.quantum import parity

if TYPE_CHECKING:
    from pennylane.wires import Wires


class BasicFiltering(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        op_list = []
        params1, params2 = params.reshape((2, len(wires), 3))

        # First Rot layer
        op_list += [qml.Rot(*angles, wire) for angles, wire in zip(params1, wires)]

        # CNOT gates
        control, target = tee(wires)
        first = next(target, None)
        op_list += [qml.CNOT(wires=cnot_wires) for cnot_wires in zip(control, target)]

        # Final CNOT gate (last to first)
        if len(wires) > 1:
            op_list += [qml.CNOT(wires=(wires[-1], first))]

        # Second Rot layer
        op_list += [qml.Rot(*angles, wire) for angles, wire in zip(params2, wires)]

        return op_list

    @staticmethod
    def _shape(wires: Wires) -> int:
        return 6 * len(wires)


class BasicFiltering2(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        op_list = []
        # params1, params2 = params.reshape((2, len(wires), 3))

        # First Rot layer
        # op_list += [qml.Rot(*angles, wire) for angles, wire in zip(params1, wires)]

        # CNOT gates
        control, target = tee(wires)
        first = next(target, None)
        op_list += [qml.CNOT(wires=cnot_wires) for cnot_wires in zip(control, target)]

        # Final CNOT gate (last to first)
        if len(wires) > 1:
            op_list += [qml.CNOT(wires=(wires[-1], first))]

        # Second Rot layer
        op_list += [qml.RY(theta, wire) for theta, wire in zip(params, wires)]

        return op_list

    @staticmethod
    def _shape(wires: Wires) -> int:
        return len(wires)


class BasicFiltering3(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        op_list = []
        params = params.reshape((len(wires) + 1, 3))

        # First Rot layer
        op_list += [qml.Rot(*angles, wire) for angles, wire in zip(params, wires)]

        # CNOT gates
        control, target = tee(wires)
        first = next(target, None)
        op_list += [qml.CNOT(wires=cnot_wires) for cnot_wires in zip(control, target)]

        # Final CNOT gate (last to first)
        if len(wires) > 1:
            op_list += [qml.CNOT(wires=(wires[-1], first))]

        # Second Rot layer
        op_list += [qml.Rot(*params[-1], first)]

        return op_list

    @staticmethod
    def _shape(wires: Wires) -> int:
        return 3 * (len(wires) + 1)


class BasicFiltering4(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        op_list = []

        # CNOT gates
        control, target = tee(wires)
        first = next(target, None)
        op_list += [qml.CNOT(wires=cnot_wires) for cnot_wires in zip(control, target)]

        # Final CNOT gate (last to first)
        if len(wires) > 1:
            op_list += [qml.CNOT(wires=(wires[-1], first))]

        # Second Rot layer
        op_list += [qml.Rot(*params, first)]

        return op_list

    @staticmethod
    def _shape(wires: Wires) -> int:
        return 3


class BasicFiltering5(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        op_list = []
        theta1, phi1, theta2, phi2 = params.reshape((4, len(wires)))

        # First Rot layer
        op_list += [qml.RX(phi, wire) for phi, wire in zip(phi1, wires)]
        op_list += [qml.RY(theta, wire) for theta, wire in zip(theta1, wires)]

        # CNOT gates
        control, target = tee(wires)
        first = next(target, None)
        op_list += [qml.CNOT(wires=cnot_wires) for cnot_wires in zip(control, target)]

        # Final CNOT gate (last to first)
        if len(wires) > 1:
            op_list += [qml.CNOT(wires=(wires[-1], first))]

        # Second Rot layer
        op_list += [qml.RY(theta, wire) for theta, wire in zip(theta2, wires)]
        op_list += [qml.RX(phi, wire) for phi, wire in zip(phi2, wires)]

        return op_list

    @staticmethod
    def _shape(wires: Wires) -> int:
        return 4 * len(wires)


class BasicPooling(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        wires, ctrl, *_ = np.split(wires, [-1])
        # return [qml.cond(qml.measure(ctrl) == 0, SimpleFiltering)(params, wires)]
        return qml.ctrl(BasicFiltering, ctrl)(params, wires)

    @staticmethod
    def _shape(wires: Wires) -> int:
        return 6 * (len(wires) - 1)


class BasicAnsatz(Ansatz):
    num_classes: int = 2
    convolve: type[Operation] = BasicFiltering
    pool: type[Operation] = BasicPooling
    fully_connected: type[Operation] = BasicFiltering

    def circuit(self, *params):
        (params,) = params
        max_wires = np.cumsum(self.num_qubits)
        offset = -int(
            np.log2(self.num_classes) // -len(self.num_qubits)
        )  # Ceiling division
        wires = max_wires - offset

        # Final behavior has different behavior
        for i in reversed(range(1, self.num_layers)):
            # Apply convolution layer:
            idx = [self.convolve.shape(wires)]
            conv_params, params = np.split(params, idx)
            self.convolve(conv_params, wires - i)

            # Apply pooling layers
            for j, (target_wire, max_wire) in enumerate(zip(wires, max_wires)):
                pool_wires = list(range(target_wire - i + 1, max_wire)) + [
                    target_wire - i
                ]
                idx = [self.pool.shape(pool_wires)]
                pool_params, params = np.split(params, idx)
                self.pool(pool_params, pool_wires)

        # Qubits to measure
        meas = np.array(
            [
                np.arange(target_wire, max_wire)
                for target_wire, max_wire in zip(wires, max_wires)
            ]
        ).flatten(order="F")

        # Final layer of convolution
        self.fully_connected(params[: self.fully_connected.shape(meas)], np.sort(meas))

        # Return the minimum required number of qubits to measure in order
        # return np.sort(meas[: int(np.ceil(np.log2(num_classes)))])
        return np.sort(meas)

    def post_processing(self, result):
        result = super().post_processing(result)

        return parity(result)

    @property
    def shape(self):
        """
        This formula was calculated pre-refactoring and I'm too lazy to recompute it
        """
        n_conv_params = self.convolve.shape(self.num_qubits) * self.num_layers
        n_pool_params = int(
            self.pool.shape(range(2))
            * len(self.num_qubits)
            * (self.num_layers - 1)
            * (
                self.num_layers / 2
                - (np.log2(self.num_classes) // -len(self.num_qubits))
                - 1
            )
        )

        return n_conv_params + n_pool_params

    @property
    def max_layers(self) -> int:
        return (
            1
            + min(self.num_qubits)
            + int(np.log2(self.num_classes) // -len(self.num_qubits))
        )
