import numpy as np
from itertools import tee
import pennylane as qml

# from pennylane.templates import BasicEntanglerLayers
from qcc.quantum.operation.ansatz import Ansatz
from qcc.quantum.operation import Unitary
from qcc.quantum import parity


class FullyConnectedLayer(Unitary):
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

        return op_list

    @staticmethod
    def _shape(wires) -> int:
        return 3 * len(wires)


class FullyConnectedAnsatz(Ansatz):
    layer = FullyConnectedLayer
    # @property
    # def _shape(self):
    #     return BasicEntanglerLayers.shape(self.num_layers, self.qubits.total)

    def circuit(self, *params):
        (params,) = params
        params = params.reshape((self.num_layers, self.layer.shape(self.qubits.total)))
        # BasicEntanglerLayers(params, wires=self.qubits.flatten())
        for p in params:
            self.layer(p, wires=self.qubits.flatten())

        return self.qubits.flatten()

    def post_processing(self, result):
        result = super().post_processing(result)

        return parity(result)

    @Ansatz.parameter  # pylint: disable=no-member
    def shape(self):
        # return np.prod(self._shape)
        return self.num_layers * self.layer.shape(self.qubits.total)

    @property
    def max_layers(self) -> int:
        return np.inf
