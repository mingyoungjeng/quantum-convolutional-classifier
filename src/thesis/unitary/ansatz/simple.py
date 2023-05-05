from unitary.simple import SimpleConvolution, SimplePooling
from ansatz.ansatz import ConvolutionAnsatz
from itertools import zip_longest, tee
import numpy as np


# TODO: work with num_classes > 2
class BaselineAnsatz(ConvolutionAnsatz):
    convolve = SimpleConvolution()
    pool = SimplePooling()

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def __call__(self, params, *_, num_layers: int = None, **__):
        max_wires = np.cumsum(dims_q)
        offset = -int(np.log2(num_classes) // -len(dims_q))  # Ceiling division
        wires = max_wires - offset

        # Final behavior has different behavior
        for i in reversed(range(1, num_layers)):
            # Apply convolution layer:
            params = self.convolution(params, wires - i)

            # Apply pooling layers
            for j, (target_wire, max_wire) in enumerate(zip(wires, max_wires)):
                p = self.pooling(
                    params, target_wire - i, range(target_wire - i + 1, max_wire)
                )
            params = p

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
        n_conv_params = self.convolve.total_params() * num_layers
        n_pool_params = self.pool.total_params() * (num_layers - 1)

        return n_conv_params + n_pool_params

    @property
    def max_layers(self) -> int:
        return (
            1 + min(self.dims_q) + int(np.log2(len(self.classes)) // -len(self.dims_q))
        )
