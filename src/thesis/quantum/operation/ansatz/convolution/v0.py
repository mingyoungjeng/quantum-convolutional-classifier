from __future__ import annotations
from typing import TYPE_CHECKING

import pennylane as qml
from pennylane.wires import Wires
from thesis.quantum import to_qubits, parity
from thesis.quantum.operation import Shift
from thesis.quantum.operation.ansatz import Ansatz
from thesis.quantum.operation.ansatz.basic import BasicFiltering
from thesis.quantum.operation.ansatz.simple import SimpleFiltering

if TYPE_CHECKING:
    from typing import Iterable
    from numbers import Number
    from thesis.quantum.operation import Unitary, Parameters, Qubits


class ConvolutionAnsatz(Ansatz):
    U_filter: Unitary = BasicFiltering
    U_fully_connected: Unitary = SimpleFiltering
    filter_shape: Iterable[int] = (2, 2)
    stride: int = 1

    @property
    def filter_shape_qubits(self):
        return to_qubits(self.filter_shape)

    def conv_pool(self, params, qubits) -> Qubits:
        fltr_ndim = len(self.filter_shape)
        fsq = self.filter_shape_qubits

        # Add extra dimensions if necessary
        # TODO: Find a prettier way of doing this
        qubits += [[] for _ in range(fltr_ndim + self.ndim - len(qubits))]

        # Shift operation
        for i, fq in enumerate(fsq):
            if len(qubits[i]) > fq:
                ctrl_qubits, img_qubits = qubits[i][:fq], qubits[i][fq:]
                qubits[i] = ctrl_qubits + img_qubits[fq:]
                qubits[self.ndim + i] += img_qubits[:fq]
            else:
                qubits[self.ndim + i] += qubits[i]
                ctrl_qubits = qubits[i] = []

            for j, control_qubit in enumerate(ctrl_qubits):
                qml.ctrl(Shift, control_qubit)(-self.stride, wires=img_qubits[j:])

        # Apply convolution filter
        filter_qubits = [q for w, fq in zip(qubits[self.ndim :], fsq) for q in w[-fq:]]
        self.U_filter(params, filter_qubits)

        return qubits

    def circuit(self, params: Parameters) -> Wires:
        qubits = self.qubits.copy()
        n_params = self.U_filter.shape(range(sum(self.filter_shape_qubits)))

        # Hybrid convolution/pooling layers
        for _ in range(self.num_layers):
            conv_params, params = params[:n_params], params[n_params:]
            qubits = self.conv_pool(conv_params, qubits)

        # Flatten wires (1D now)
        # TODO: Find a more elegant solution maybe?
        wires = [q for w in qubits for q in w]

        # Fully connected layer
        self.U_fully_connected(params, wires)

        return wires[::-1]

    def post_processing(self, result) -> Iterable[Iterable[Number]]:
        result = super().post_processing(result)

        return parity(result)

    @property
    def shape(self) -> int:
        n_params = self.num_layers * self.U_filter.shape(
            range(sum(self.filter_shape_qubits))
        )
        n_params += self.U_fully_connected.shape(self.wires)

        return n_params

    @property
    def max_layers(self) -> int:
        return self.min_dim
