from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import Controlled
from pennylane.wires import Wires

from qcc.quantum import to_qubits, wires_to_qubits
from qcc.quantum.pennylane import Shift, C2Q, Qubits

if TYPE_CHECKING:
    from typing import Iterable

    ConvolutionSetting = int | Iterable[int]


# TODO: stride and padding and dilation currently aren't implemented
class Convolution(Operation):
    """Convolution operation in Pennylane"""

    num_wires = AnyWires

    def __init__(
        self,
        params,
        qubits,
        # stride: ConvolutionSetting = 1,
        # padding: ConvolutionSetting = 0,
        # dilation: ConvolutionSetting = 1,
        do_swaps=True,
        id=None,
    ):
        dims_q = [len(q) for q in qubits]

        self._hyperparameters = {
            "dims_q": dims_q,
            # "stride": stride,
            # "padding": padding,
            # "dilation": dilation,
            "do_swaps": do_swaps,
        }

        wires = Wires.all_wires(qubits)
        super().__init__(params, wires, id=id)

    @property
    def dims_q(self) -> ConvolutionSetting:
        return self.hyperparameters["dims_q"]

    @property
    def stride(self) -> ConvolutionSetting:
        return 1  # self.hyperparameters["stride"]

    # @property
    # def padding(self) -> ConvolutionSetting:
    #     return self.hyperparameters["padding"]

    # @property
    # def dilation(self) -> ConvolutionSetting:
    #     return self.hyperparameters["dilation"]

    @staticmethod
    def shift(kernel_shape_q, qubits, stride=1, H=True):
        op_list = []

        for i, fsq in enumerate(kernel_shape_q):
            data_wires = qubits[i]
            filter_wires = qubits[i - len(kernel_shape_q)][:fsq]

            if len(data_wires) == 0:
                continue

            # Apply Hadamard to kernel wires
            if H:
                op_list += [qml.Hadamard(aq) for aq in filter_wires]

            # Shift operation
            op_list += [
                Controlled(Shift(-stride, wires=data_wires[j:]), control)
                for j, control in enumerate(filter_wires)
            ]

        return op_list

    @staticmethod
    def filter(kernel: np.ndarray, qubits):
        qubits = Qubits(q[:fsq] for q, fsq in zip(qubits, to_qubits(kernel.shape)))

        return [C2Q(kernel, wires=qubits.flatten(), transpose=True)]

    @staticmethod
    def permute(kernel_shape_q, qubits):
        op_list = []

        for i, fsq in enumerate(kernel_shape_q):
            data_wires = qubits[i][:fsq]
            filter_wires = qubits[i - len(kernel_shape_q)][:fsq]

            op_list += [qml.SWAP((f, a)) for f, a in zip(data_wires, filter_wires)]

        return op_list

    @staticmethod
    def compute_decomposition(
        *params,
        wires: Wires,
        **hyperparameters,
    ) -> Iterable[Operation]:
        # Keep the type-checker happy
        (params,) = params

        dims_q = hyperparameters["dims_q"]
        do_swaps = hyperparameters["do_swaps"]

        kernel_shape_q = to_qubits(params.shape)
        qubits = wires_to_qubits(dims_q, wires)

        op_list = Convolution.shift(kernel_shape_q, qubits)

        if kernel_shape_q.any():
            op_list += Convolution.filter(params, qubits)

        if do_swaps:
            op_list += Convolution.permute(kernel_shape_q, qubits)

        return op_list
