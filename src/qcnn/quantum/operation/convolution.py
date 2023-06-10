from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import Controlled
from pennylane.wires import Wires

from qcnn.quantum import to_qubits, wires_to_qubits
from qcnn.quantum.operation import Shift, C2Q

if TYPE_CHECKING:
    from typing import Iterable

    ConvolutionSetting = int | Iterable[int]


# TODO: stride and padding and dilation currently aren't implemented
class Convolution(Operation):
    num_wires = AnyWires

    def __init__(
        self,
        params,
        qubits,
        # stride: ConvolutionSetting = 1,
        # padding: ConvolutionSetting = 0,
        # dilation: ConvolutionSetting = 1,
        do_swaps=True,
        do_queue=True,
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
        super().__init__(params, wires, do_queue=do_queue, id=id)

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
    def shift(filter_shape_q, qubits, stride=1, H=True):
        op_list = []

        for i, fsq in enumerate(filter_shape_q):
            filter_wires = qubits[i]
            ancilla_wires = qubits[i - len(filter_shape_q)][:fsq]

            # Apply Hadamard to ancilla wires
            if H:
                op_list += [qml.Hadamard(aq) for aq in ancilla_wires]

            # Shift operation
            op_list += [
                Controlled(Shift(-stride, wires=filter_wires[j:]), control)
                for j, control in enumerate(ancilla_wires)
            ]

        return op_list

    @staticmethod
    def filter(fltr: np.ndarray, qubits):
        wires = [q[:fsq] for q, fsq in zip(qubits, to_qubits(fltr.shape))]
        wires = Wires.all_wires(wires)

        return [C2Q(fltr, wires=wires, transpose=True)]

    @staticmethod
    def permute(filter_shape_q, qubits):
        op_list = []

        for i, fsq in enumerate(filter_shape_q):
            filter_wires = qubits[i][:fsq]
            ancilla_wires = qubits[i - len(filter_shape_q)][:fsq]

            op_list += [qml.SWAP((f, a)) for f, a in zip(filter_wires, ancilla_wires)]

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

        filter_shape_q = to_qubits(params.shape)
        qubits = wires_to_qubits(dims_q, wires)

        op_list = Convolution.shift(filter_shape_q, qubits)

        if filter_shape_q.any():
            op_list += Convolution.filter(params, qubits)

        if do_swaps:
            op_list += Convolution.permute(filter_shape_q, qubits)

        return op_list
