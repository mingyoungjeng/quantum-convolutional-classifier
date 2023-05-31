from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

import pennylane as qml
from pennylane.wires import Wires
from pennylane.templates import AmplitudeEmbedding

from torch.nn import Module
from thesis.quantum import to_qubits, wires_to_qubits
from thesis.ml import is_iterable

if TYPE_CHECKING:
    from typing import Iterable, Optional
    from numbers import Number
    from pennylane.operation import Operation
    from thesis.quantum.operation import Parameters, Qubits

    Statevector = Iterable[Number]


def is_multidimensional(wires: Qubits):
    if is_iterable(wires):
        return any(is_iterable(w) for w in wires)
    return False


# TODO: turn into metaclass / decorator
# (especially useful for post_init parameters initialization)
class Ansatz(Module, ABC):
    __slots__ = "_qubits", "_num_layers"

    def __init__(self, qubits, num_layers=None):
        super().__init__()
        self.qubits = qubits
        self.num_layers = self.max_layers if num_layers is None else num_layers

    @property
    def qubits(self) -> Iterable[Iterable[int]]:
        return self._qubits

    @qubits.setter
    def qubits(self, q) -> None:
        self._qubits = [Wires(w) for w in q] if is_multidimensional(q) else [Wires(q)]

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value: int) -> None:
        # Check bounds
        if value < 0 or value > self.max_layers:
            err = f"num_layers must be in range (0, {self.max_layers}], got {value}"
            raise ValueError(err)

        self._num_layers = value

    # Derived properties

    @property
    def qnode(self) -> qml.QNode:
        wires = self.wires[::-1]  # Big-endian format
        device = qml.device("default.qubit", wires=wires)
        return qml.QNode(self._circuit, device, interface="torch")

    @property
    def wires(self) -> Wires:
        return Wires.all_wires(self.qubits)

    @property
    def num_wires(self) -> int:
        return len(self.wires)

    @property
    def num_qubits(self) -> Iterable[int] | int:
        return [len(q) for q in self.qubits]

    @property
    def ndim(self) -> int:
        return len(self.num_qubits)

    @property
    def min_dim(self) -> int:
        return min(self.num_qubits)

    # Circuit operation

    def c2q(self, psi_in: Statevector) -> Operation:
        return AmplitudeEmbedding(psi_in, self.wires[::-1], pad_with=0, normalize=True)

    def q2c(self, wires: Wires):
        return qml.probs(wires)

    def post_processing(self, result) -> Iterable[Iterable[Number]]:
        # Makes sure batch is 2D array
        if result.dim() == 1:
            result = result.unsqueeze(0)

        return result

    def _circuit(
        self,
        psi_in: Optional[Statevector] = None,
        params: Optional[Parameters] = None,
    ):
        if psi_in is not None:  # this is done to facilitate drawing
            self.c2q(psi_in)
        meas = self.circuit(next(self.parameters()) if params is None else params)
        return self.q2c(meas)

    def forward(self, psi_in: Optional[Statevector] = None):
        result = self.qnode(psi_in=psi_in)  # pylint: disable=not-callable

        return self.post_processing(result)

    # Abstract methods

    @abstractmethod
    def circuit(self, params: Parameters) -> Wires:
        pass

    @property
    @abstractmethod
    def shape(self) -> int:
        # TODO: for now, parameters are set using self.shape(), but want to change that
        pass

    @property
    @abstractmethod
    def max_layers(self) -> int:
        pass

    # Instance factories

    @classmethod
    def from_dims(cls, dims: Iterable[int], *args, **kwargs):
        dims_q = to_qubits(dims)
        qubits = wires_to_qubits(dims_q)

        ansatz = cls(qubits, *args, **kwargs)

        return ansatz
