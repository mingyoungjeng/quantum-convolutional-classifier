from itertools import tee
import numpy as np
import pennylane as qml
from pennylane.operation import Operation
from thesis.operation.ansatz import Ansatz
from thesis.operation import Unitary


class NaiveConvolution(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        op_list = []
        params = params.reshape((len(wires), 3))

        # First Rot layer
        op_list += [qml.Rot(*angles, wire) for angles, wire in zip(params, wires)]

        # CNOT gates
        control, target = tee(wires)
        first = next(target, None)
        op_list += [qml.CNOT(wires=cnot_wires) for cnot_wires in zip(control, target)]

        # Final CNOT gate (last to first)
        if len(wires) > 1:
            op_list += [qml.CNOT(wires=(wires[-1], first))]

    @staticmethod
    def shape(*_) -> int:
        return 3


class NaiveAnsatz(Ansatz):
    convolve: type[Operation] = NaiveConvolution
    fully_connected: type[Operation] = NaiveConvolution

    def circuit(self, params):
        max_wires = np.cumsum(self.num_qubits)

        idx = [self.convolve.shape() * len(self.num_qubits)]
        for i in range(self.num_layers):
            # Apply convolution layers
            for j in 1 + np.arange(min(self.num_qubits)):
                conv_params, params = np.split(params, idx)
                self.convolve(conv_params, max_wires - j)

        idx = [self.convolve.shape() * len(self.wires)]
        conv_params, params = np.split(params, idx)
        self.convolve(conv_params, self.wires)

        params, _ = np.split(params, idx)
        params = params.reshape((len(self.wires), 3))
        for (theta, phi, delta), wire in zip(params, self.wires):
            qml.Rot(theta, phi, delta, wires=wire)

        return self.wires

    @property
    def shape(self):
        n_params = (
            self.convolve.shape()
            * len(self.num_qubits)
            * min(self.num_qubits)
            * self.num_layers
        )
        n_params += self.convolve.shape() * self.num_wires * 2

        return n_params

    @property
    def max_layers(self) -> int:
        return min(self.num_qubits)
