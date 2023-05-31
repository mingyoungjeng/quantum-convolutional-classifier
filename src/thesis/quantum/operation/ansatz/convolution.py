from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from pennylane.wires import Wires
from pennylane.templates import AmplitudeEmbedding

from thesis.quantum import to_qubits
from thesis.quantum.operation.ansatz import Ansatz
from thesis.quantum.operation import Convolution
from thesis.quantum.operation.ansatz.simple import SimpleConvolution
from thesis.ml.optimize import init_params

if TYPE_CHECKING:
    from typing import Iterable
    from thesis.quantum.operation import Unitary, Parameters, Qubits


class ConvolutionAnsatz(Ansatz):
    U_fully_connected: Unitary = SimpleConvolution
    filter_shape: Iterable[int] = (2, 2)
    stride: int = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._params = init_params(self.shape)

    @property
    def filter_shape_qubits(self):
        return to_qubits(self.filter_shape)

    @property
    def main_qubits(self):
        return self.qubits[: -(self.num_layers * len(self.filter_shape))]

    @property
    def main_wires(self):
        return Wires.all_wires(self.main_qubits)

    @property
    def ancilla_qubits(self):
        return self.qubits[-(self.num_layers * len(self.filter_shape)) :]

    @property
    def ancilla_wires(self):
        return Wires.all_wires(self.ancilla_qubits)

    def c2q(self, psi_in):
        return AmplitudeEmbedding(
            psi_in, self.main_wires[::-1], pad_with=0, normalize=True
        )

    def circuit(self, params: Parameters) -> Wires:
        n_params = np.prod(self.filter_shape)

        # Hybrid convolution/pooling layers
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

        return self.wires

    @property
    def shape(self) -> int:
        n_params = self.num_layers * np.prod(self.filter_shape)
        n_params += self.U_fully_connected.shape(self.wires)

        return n_params

    @property
    def max_layers(self) -> int:
        return self.min_dim

    @classmethod
    def from_dims(cls, *args, **kwargs):
        self = super().from_dims(*args, **kwargs)

        # Add ancilla qubits
        ancilla_qubits = []
        top = self.num_wires
        for _ in range(self.num_layers):
            for fsq in self.filter_shape_qubits:
                ancilla_qubits += [list(range(top, top + fsq))]
                top += fsq

        self.qubits = self.qubits + ancilla_qubits

        return self
