from __future__ import annotations
from typing import TYPE_CHECKING

from attrs import define
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from thesis.quantum import to_qubits
from thesis.operation import Shift
from thesis.operation.ansatz import Ansatz, is_multidimensional
from thesis.operation.ansatz.basic import BasicConvolution

if TYPE_CHECKING:
    from typing import Iterable
    from thesis.operation import Unitary, Parameters, Qubits


@define(frozen=True)
class ConvolutionAnsatz(Ansatz):
    U: Unitary = BasicConvolution
    filter_shape: Iterable[int] = (2, 2)  # TODO: add type hint
    stride: int = 1

    @property
    def filter_shape_qubits(self):
        return to_qubits(self.filter_shape)

    @property
    def num_params_per_layer(self):
        return self.U.shape(range(sum(self.filter_shape_qubits)))

    def conv_pool(self, qubits, params) -> Qubits:
        ndim = len(self.wires)
        fltr_ndim = len(self.filter_shape)
        fsq = self.filter_shape_qubits

        # Add extra dimensions if necessary
        for _ in range(fltr_ndim + ndim - len(qubits)):
            qubits += [[]]

        # Shift operation
        for i, fq in enumerate(fsq):
            if len(qubits[i]) > fq:
                ctrl_qubits, img_qubits = qubits[i][:fq], qubits[i][fq:]
                qubits[i] = ctrl_qubits + img_qubits[fq:]
                qubits[ndim + i] += img_qubits[:fq]
            else:
                qubits[ndim + i] += qubits[i]
                ctrl_qubits = qubits[i] = []

            for j, control_qubit in enumerate(ctrl_qubits):
                qml.ctrl(Shift, control_qubit)(-self.stride, wires=img_qubits[j:])

        # Apply convolution filter
        filter_qubits = [q for w, fq in zip(qubits[ndim:], fsq) for q in w[-fq:]]
        self.U(params, filter_qubits)

        return qubits

    def circuit(self, params: Parameters) -> Wires:
        qubits = self.qubits.copy()

        # TODO: Find a more elegant solution maybe?
        if not is_multidimensional(qubits):
            qubits = [qubits]

        n_params = self.num_params_per_layer

        # Hybrid convolution/pooling layers
        for _ in range(self.num_layers):
            conv_params, params = params[:n_params], params[n_params:]
            qubits = self.conv_pool(qubits, conv_params)

        # Flatten wires (1D now)
        # TODO: Find a more elegant solution maybe?
        wires = [q for w in qubits for q in w]

        # Fully connected layer
        self.U(params, wires)

        return wires

    @property
    def shape(self) -> int:
        n_params = self.num_layers * self.num_params_per_layer
        n_params += self.U.shape(self.wires)

        return n_params

    @property
    def max_layers(self) -> int:
        # TODO: check to see if this works 1D
        return np.min(self.num_qubits)
