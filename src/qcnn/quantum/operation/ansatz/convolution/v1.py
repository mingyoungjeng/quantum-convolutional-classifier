from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from pennylane.wires import Wires

from qcnn.quantum.operation import Convolution
from qcnn.quantum.operation.ansatz.convolution.test import ConvolutionAnsatz as Base

if TYPE_CHECKING:
    from qcnn.quantum.operation import Parameters


class ConvolutionAnsatz(Base):
    filter_shape = (4, 4)

    def circuit(self, params: Parameters) -> Wires:
        n_params = np.prod(self.filter_shape)

        # Convolution layers
        for i in range(self.num_layers):
            conv_params, params = params[:n_params], params[n_params:]
            conv_params = conv_params.reshape(self.filter_shape)

            qubits = (
                self.main_qubits
                + self.ancilla_qubits[
                    len(self.filter_shape) * i : len(self.filter_shape) * i
                    + len(self.filter_shape)
                ]
            )

            Convolution(conv_params, qubits)

        # Fully connected layer
        self.U_fully_connected(params, self.wires)

        return self.wires[::-1]

    @property
    def shape(self) -> int:
        n_params = self.num_layers * np.prod(self.filter_shape)
        n_params += self.U_fully_connected.shape(self.wires)

        return n_params

    @property
    def max_layers(self) -> int:
        return self.min_dim
