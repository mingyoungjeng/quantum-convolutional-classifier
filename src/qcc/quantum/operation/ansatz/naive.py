import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from qcc.quantum.operation.ansatz import Ansatz
from qcc.quantum.operation.ansatz.fully_connected import FullyConnectedLayer
from qcc.quantum import parity


class NaiveAnsatz(Ansatz):
    convolve: type[Operation] = FullyConnectedLayer
    fully_connected: type[Operation] = FullyConnectedLayer

    def circuit(self, *params):
        (params,) = params
        max_wires = np.cumsum(self.qubits.shape)

        idx = [self.convolve.shape(len(self.qubits.shape))]
        for i in range(self.num_layers):
            # Apply convolution layers
            for j in 1 + np.arange(min(self.qubits.shape)):
                conv_params, params = np.split(params, idx)
                self.convolve(conv_params, max_wires - j)

        idx = [self.convolve.shape(len(self.qubits.flatten()))]
        conv_params, params = np.split(params, idx)
        self.convolve(conv_params, self.qubits.flatten())

        params, _ = np.split(params, idx)
        params = params.reshape((len(self.qubits.flatten()), 3))
        for (theta, phi, delta), wire in zip(params, self.qubits.flatten()):
            qml.Rot(theta, phi, delta, wires=wire)

        return self.qubits.flatten()

    def post_processing(self, result):
        result = super().post_processing(result)

        return parity(result)

    @Ansatz.parameter  # pylint: disable=no-member
    def shape(self):
        n_params = (
            self.convolve.shape(len(self.qubits.shape) * min(self.qubits.shape))
            * self.num_layers
        )
        n_params += self.convolve.shape(self.qubits.total) * 2

        return n_params

    @property
    def max_layers(self) -> int:
        return min(self.qubits.shape)
