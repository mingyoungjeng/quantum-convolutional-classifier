from __future__ import annotations
from typing import TYPE_CHECKING

from torch.nn import Module
from pennylane.wires import Wires
from pennylane.templates import AmplitudeEmbedding as Initialize

from qcnn.quantum import to_qubits, parity
from qcnn.quantum.operation.ansatz import Ansatz, is_multidimensional
from qcnn.quantum.operation.ansatz.fully_connected import FullyConnectedLayer
from qcnn.ml.optimize import init_params

if TYPE_CHECKING:
    from typing import Iterable
    from numbers import Number
    from qcnn.quantum.operation import Unitary, Parameters


class ConvolutionAnsatz(Ansatz):
    __slots__ = "_ancilla_qubits"

    U_fully_connected: Unitary = FullyConnectedLayer
    filter_shape: Iterable[int] = (2, 2)
    stride: int = 1

    def __init__(self, qubits, num_layers=None):
        Module.__init__(self)
        self.main_qubits = qubits
        self._ancilla_qubits = []
        self.num_layers = num_layers

        # Add ancilla qubits if None
        top = self.num_wires
        for _ in range(self.num_layers):
            for fsq in self.filter_shape_qubits:
                self.ancilla_qubits += [list(range(top, top + fsq))]
                top += fsq

        self._params = init_params(self.shape)

    @property
    def qubits(self) -> Iterable[Iterable[int]]:
        return self.main_qubits + self.ancilla_qubits

    @property
    def filter_shape_qubits(self):
        return to_qubits(self.filter_shape)

    @property
    def main_qubits(self):
        return self._qubits.copy()

    @main_qubits.setter
    def main_qubits(self, q) -> None:
        self._qubits = [Wires(w) for w in q] if is_multidimensional(q) else [Wires(q)]

    @property
    def main_wires(self):
        return Wires.all_wires(self.main_qubits)

    @property
    def ancilla_qubits(self):
        return self._ancilla_qubits.copy()

    @ancilla_qubits.setter
    def ancilla_qubits(self, q) -> None:
        self._ancilla_qubits = (
            [Wires(w) for w in q] if is_multidimensional(q) else [Wires(q)]
        )

    @property
    def ancilla_wires(self):
        return Wires.all_wires(self.ancilla_qubits)

    @property
    def ndim(self) -> int:
        return len(self.main_qubits)

    @property
    def min_dim(self) -> int:
        return min((len(q) for q in self.main_qubits))

    def c2q(self, psi_in):
        return Initialize(psi_in, self.main_wires[::-1], pad_with=0, normalize=True)

    # def post_processing(self, result) -> Iterable[Iterable[Number]]:
    #     result = super().post_processing(result)

    #     return parity(result, self.num_classes)
