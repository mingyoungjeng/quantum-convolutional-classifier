from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import tee

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import Controlled, QubitUnitary
from pennylane.wires import Wires

from qcc.quantum import to_qubits, wires_to_qubits
from qcc.quantum.pennylane import Shift, C2Q, Qubits, Unitary

if TYPE_CHECKING:
    from typing import Iterable

    ConvolutionSetting = int | Iterable[int]


def define_filter(op: type[Operation] = qml.RY, num_layers: int = 1):
    class Wrapper(FilterQML):
        def __init__(self, *params, wires: Wires, id=None, **_):
            super().__init__(*params, wires=wires, op=op, num_layers=num_layers, id=id)

        @staticmethod
        def _shape(num_wires: int, **_):
            return FilterQML._shape(num_wires, op=op, num_layers=num_layers)

    return Wrapper


class FilterQML(Unitary):
    """Alternative to C2Q for implementing convolutional filters in QML"""

    def __init__(
        self,
        *params,
        wires: Wires,
        op: type[Operation] = qml.RY,
        num_layers: int = 1,
        id=None,
        **_,
    ):
        self._hyperparameters = {"op": op, "num_layers": num_layers}

        super().__init__(*params, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(*params, wires, **hyperparameters):
        (params,) = params
        op: type[Operation] = hyperparameters["op"]
        num_layers: int = hyperparameters["num_layers"]

        op_list = []
        shape = (num_layers, len(wires), op.num_params)
        params = params.reshape(shape)

        for i, angles in enumerate(params):
            # Rotation gate layer
            op_list += [op(*a, wire) for a, wire in zip(angles, wires)]

            if i >= len(params) - 1:
                continue

            # CNOT gates
            # wires = wires[::-1]
            control, target = tee(wires)
            first = next(target, None)
            op_list += [qml.CNOT(wires=w) for w in zip(control, target)]

            # Final CNOT gate (last to first)
            if len(wires) > 1:
                op_list += [qml.CNOT(wires=(wires[-1], first))]

        return op_list

    @staticmethod
    def _shape(num_wires: Wires, **hyperparameters) -> int:
        op: type[Operation] = hyperparameters["op"]
        num_layers: int = hyperparameters["num_layers"]

        return num_layers * num_wires * op.num_params


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
    def shift(filter_shape_q, qubits, stride=1, H=True):
        op_list = []

        for i, fsq in enumerate(filter_shape_q):
            data_wires = qubits[i]
            filter_wires = qubits[i - len(filter_shape_q)][:fsq]

            if len(data_wires) == 0:
                continue

            # Apply Hadamard to ancilla wires
            if H:
                op_list += [qml.Hadamard(aq) for aq in filter_wires]

            # Shift operation
            op_list += [
                Controlled(Shift(-stride, wires=data_wires[j:]), control)
                for j, control in enumerate(filter_wires)
            ]

        return op_list

    @staticmethod
    def filter(fltr: np.ndarray, qubits):
        qubits = Qubits(q[:fsq] for q, fsq in zip(qubits, to_qubits(fltr.shape)))

        return [C2Q(fltr, wires=qubits.flatten(), transpose=True)]

    @staticmethod
    def permute(filter_shape_q, qubits):
        op_list = []

        for i, fsq in enumerate(filter_shape_q):
            data_wires = qubits[i][:fsq]
            filter_wires = qubits[i - len(filter_shape_q)][:fsq]

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

        filter_shape_q = to_qubits(params.shape)
        qubits = wires_to_qubits(dims_q, wires)

        op_list = Convolution.shift(filter_shape_q, qubits)

        if filter_shape_q.any():
            op_list += Convolution.filter(params, qubits)

        if do_swaps:
            op_list += Convolution.permute(filter_shape_q, qubits)

        return op_list
