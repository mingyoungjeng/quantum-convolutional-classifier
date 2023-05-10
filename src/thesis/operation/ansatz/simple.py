from itertools import tee
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from thesis.operation.ansatz import Ansatz
from thesis.operation import Unitary


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


class SimplePooling(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        wires, ctrl, *_ = np.split(wires, [-1])
        # return [qml.cond(qml.measure(ctrl) == 0, SimpleConvolution)(params, wires)]
        return qml.ctrl(SimpleConvolution, ctrl)(params, wires)

    @staticmethod
    def shape(*_) -> int:
        return 6


class SimpleAnsatz(Ansatz):
    num_classes: int = 2
    convolve: type[Operation] = SimpleConvolution
    pool: type[Operation] = SimplePooling
    fully_connected: type[Operation] = SimpleConvolution

    def circuit(self, params):
        max_wires = np.cumsum(self.num_qubits)
        offset = -int(
            np.log2(self.num_classes) // -len(self.num_qubits)
        )  # Ceiling division
        wires = max_wires - offset

        # Final behavior has different behavior
        for i in reversed(range(1, self.num_layers)):
            # Apply convolution layer:
            idx = np.cumsum([self.convolve.shape(), self.pool.shape()])
            conv, pool, params = np.split(params, idx)
            self.convolve(conv, wires - i)

            # Apply pooling layers
            for j, (target_wire, max_wire) in enumerate(zip(wires, max_wires)):
                tmp = list(range(target_wire - i + 1, max_wire)) + [target_wire - i]
                self.pool(pool, tmp)

        # Qubits to measure
        meas = np.array(
            [
                np.arange(target_wire, max_wire)
                for target_wire, max_wire in zip(wires, max_wires)
            ]
        ).flatten(order="F")

        # Final layer of convolution
        self.fully_connected(params[: self.fully_connected.shape()], np.sort(meas))

        # Return the minimum required number of qubits to measure in order
        # return np.sort(meas[: int(np.ceil(np.log2(num_classes)))])
        return np.sort(meas)

    @property
    def shape(self):
        n_conv_params = self.convolve.shape() * self.num_layers
        n_pool_params = self.pool.shape() * (self.num_layers - 1)

        return n_conv_params + n_pool_params

    @property
    def max_layers(self) -> int:
        return (
            1
            + min(self.num_qubits)
            + int(np.log2(self.num_classes) // -len(self.num_qubits))
        )
