from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from pennylane.wires import Wires

from thesis.quantum.operation.ansatz.convolution.test import ConvolutionAnsatz as Base
from thesis.quantum.operation import Convolution

import pennylane as qml
from thesis.quantum.operation import Multiplex

if TYPE_CHECKING:
    from thesis.quantum.operation import Parameters


class ConvolutionAnsatz(Base):
    def circuit(self, params: Parameters) -> Wires:
        n_params = np.prod(self.filter_shape) - 1

        # Convolution layers
        for i in range(self.num_layers):
            conv_params, params = params[:n_params], params[n_params:]
            # conv_params = conv_params.reshape(self.filter_shape)

            qubits = (
                self.main_qubits
                + self.ancilla_qubits[
                    len(self.filter_shape) * i : len(self.filter_shape) * i
                    + len(self.filter_shape)
                ]
            )

            Convolution.shift(self.filter_shape_qubits, qubits)

            ### FILTER
            wires = [q[:fsq] for q, fsq in zip(qubits, self.filter_shape_qubits)]
            wires = Wires.all_wires(wires)

            idx = lambda x: 2 ** (len(wires) - x) * (2**x - 1)
            for j, wire in enumerate(wires):
                theta = conv_params[idx(j) : idx(j + 1)]

                Multiplex(theta, wire, wires[j + 1 :], qml.RY)

            Convolution.permute(self.filter_shape_qubits, qubits)

        # Fully connected layer
        self.U_fully_connected(params, self.wires)

        return self.wires[::-1]

    @property
    def shape(self) -> int:
        n_params = self.num_layers * (np.prod(self.filter_shape) - 1)
        n_params += self.U_fully_connected.shape(self.wires)

        return n_params

    @property
    def max_layers(self) -> int:
        return self.min_dim
