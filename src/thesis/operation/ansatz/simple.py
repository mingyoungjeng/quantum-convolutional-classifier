from thesis.operation.ansatz import Ansatz
from itertools import zip_longest, tee
import numpy as np
import pennylane as qml
from thesis.operation import Unitary
from pennylane.operation import Operation


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


# TODO: work with num_classes > 2
class SimpleAnsatz(Ansatz):
    convolve: type[Operation] = SimpleConvolution()
    pool: type[Operation] = SimplePooling()

    def __call__(self, params):
        max_wires = np.cumsum(self.num_qubits)
        offset = -int(
            np.log2(self.num_classes) // -len(self.num_qubits)
        )  # Ceiling division
        wires = max_wires - offset

        # Final behavior has different behavior
        for i in reversed(range(1, self.num_layers)):
            # Apply convolution layer:
            conv_params, params = np.split(params, self.convolve.shape())
            self.convolve(conv_params, wires - i)

            # Apply pooling layers
            pool_params, params = np.split(params, self.pool.shape())
            for j, (target_wire, max_wire) in enumerate(zip(wires, max_wires)):
                self.pool(
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
        self.convolve(params, np.sort(meas))

        # Return the minimum required number of qubits to measure in order
        # return np.sort(meas[: int(np.ceil(np.log2(num_classes)))])
        return np.sort(meas)

    def total_params(self, num_layers=None, *_, **__):
        n_conv_params = self.convolve.shape() * num_layers
        n_pool_params = self.pool.shape() * (num_layers - 1)

        return n_conv_params + n_pool_params

    @property
    def max_layers(self) -> int:
        return (
            1
            + min(self.num_qubits)
            + int(np.log2(len(self.num_classes)) // -len(self.num_qubits))
        )
