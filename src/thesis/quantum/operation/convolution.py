from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires

from thesis.quantum import to_qubits, wires_to_qubits
from thesis.quantum.operation import Shift, C2Q

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
        do_queue=True,
        id=None,
    ):
        dims_q = [len(q) for q in qubits]

        self._hyperparameters = {
            "dims_q": dims_q,
            # "stride": stride,
            # "padding": padding,
            # "dilation": dilation,
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
    def _shift(filter_shape_q, qubits, stride=1):
        op_list = []

        for i, fsq in enumerate(filter_shape_q):
            filter_wires = qubits[i]
            ancilla_wires = qubits[i - len(filter_shape_q)][:fsq]

            # Apply Hadamard to ancilla wires
            op_list += [qml.Hadamard(aq) for aq in ancilla_wires]

            # Shift operation
            op_list += [
                qml.ctrl(Shift, control)(-stride, wires=filter_wires[j:])
                for j, control in enumerate(ancilla_wires)
            ]

        return op_list

    @staticmethod
    def _filter(fltr: np.ndarray, qubits):
        wires = [q[:fsq] for q, fsq in zip(qubits, to_qubits(fltr.shape))]
        wires = Wires.all_wires(wires)

        fltr = fltr.flatten("F")
        return [C2Q(fltr, wires, transpose=True)]

    @staticmethod
    def _permute(filter_shape_q, qubits):
        op_list = []

        for i, fsq in enumerate(filter_shape_q):
            filter_wires = qubits[i][:fsq]
            ancilla_wires = qubits[i - len(filter_shape_q)][:fsq]

            op_list += [qml.SWAP((f, a)) for f, a in zip(filter_wires, ancilla_wires)]

        return op_list

    @staticmethod
    def compute_decomposition(*params, wires: Wires, dims_q) -> Iterable[Operation]:
        (params,) = params  # Keep the type-checker happy

        filter_shape_q = to_qubits(params.shape)
        qubits = wires_to_qubits(dims_q, wires)

        op_list = []
        op_list += Convolution._shift(filter_shape_q, qubits)
        op_list += Convolution._filter(params, qubits)
        op_list += Convolution._permute(filter_shape_q, qubits)

        return op_list
