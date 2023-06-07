import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from qcnn.quantum.operation.ansatz import Ansatz
from qcnn.quantum.operation.ansatz.fully_connected import FullyConnectedLayer
from qcnn.quantum import parity


class NaiveAnsatz(Ansatz):
    convolve: type[Operation] = FullyConnectedLayer
    fully_connected: type[Operation] = FullyConnectedLayer

    def circuit(self, params):
        max_wires = np.cumsum(self.num_qubits)

        idx = [self.convolve.shape(len(self.num_qubits))]
        for i in range(self.num_layers):
            # Apply convolution layers
            for j in 1 + np.arange(min(self.num_qubits)):
                conv_params, params = np.split(params, idx)
                self.convolve(conv_params, max_wires - j)

        idx = [self.convolve.shape(len(self.wires))]
        conv_params, params = np.split(params, idx)
        self.convolve(conv_params, self.wires)

        params, _ = np.split(params, idx)
        params = params.reshape((len(self.wires), 3))
        for (theta, phi, delta), wire in zip(params, self.wires):
            qml.Rot(theta, phi, delta, wires=wire)

        return self.wires

    def post_processing(self, result):
        result = super().post_processing(result)

        return parity(result)

    @property
    def shape(self):
        n_params = (
            self.convolve.shape(len(self.num_qubits) * min(self.num_qubits))
            * self.num_layers
        )
        n_params += self.convolve.shape(self.num_wires) * 2

        return n_params

    @property
    def max_layers(self) -> int:
        return min(self.num_qubits)
