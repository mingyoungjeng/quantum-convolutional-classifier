from __future__ import annotations
from typing import TYPE_CHECKING

from torch import sqrt
from torch.nn.functional import pad

from pennylane import Hadamard, Select

from qcc.quantum import to_qubits
from qcc.quantum.pennylane import Qubits, QubitsProperty
from qcc.quantum.pennylane.ansatz import Ansatz
from qcc.quantum.pennylane.local import define_filter

if TYPE_CHECKING:
    from pennylane.wires import Wires
    from qcc.quantum.pennylane import Parameters, Unitary


class FullyConnected(Ansatz):
    """Fully-connected layer in Pennylane"""

    __slots__ = "_data_qubits", "_feature_qubits", "U_kernel"

    data_qubits: Qubits = QubitsProperty(slots=True)
    feature_qubits: Qubits = QubitsProperty(slots=True)

    num_classes: int

    def __init__(
        self,
        qubits: Qubits,
        num_classes: int = 2,
        q2c_method: Ansatz.Q2CMethod | str = Ansatz.Q2CMethod.Probabilities,
        U_kernel: type[Unitary] = define_filter(num_layers=4),
    ):
        self._num_layers = 1
        self.U_kernel = U_kernel  # pylint: disable=invalid-name

        # Data qubits
        self.data_qubits = Qubits(qubits)

        # Feature qubits
        top = self.data_qubits.total
        self.feature_qubits = [range(top, top + to_qubits(num_classes))]

        qubits = Qubits(self.data_qubits + self.feature_qubits)
        super().__init__(qubits, 1, num_classes, q2c_method)

    def circuit(self, *params: Parameters) -> Wires:
        (params,) = params

        for qubit in self.feature_qubits.flatten():
            Hadamard(wires=qubit)

        shape = (self.num_classes, len(params) // self.num_classes)
        params = params.reshape(shape)

        npad = 2 ** to_qubits(self.num_classes) - self.num_classes
        npad = (0, 0, 0, npad)
        params = pad(params, npad, mode="constant", value=0)

        # Apply filter
        wires = self.data_qubits.flatten()
        filters = tuple(self.U_kernel(param, wires=wires) for param in params)
        Select(filters, self.feature_qubits.flatten()[::-1])

        return Qubits(self.feature_qubits + self.data_qubits).flatten()

    @property
    def shape(self) -> int:
        return self.U_kernel.shape(self.data_qubits.total) * self.num_classes

    def _forward(self, result):
        # Get subset of output
        norm = 2**self.qubits.total
        result = norm * result[:, : self.num_classes]
        result = sqrt(result)

        return result

    @property
    def max_layers(self) -> int:
        return 1

    # ==== private ==== #

    @property
    def _data_wires(self) -> Wires:
        return self.data_qubits.flatten()
