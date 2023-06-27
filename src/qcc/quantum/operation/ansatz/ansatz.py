from __future__ import annotations
from typing import TYPE_CHECKING

from abc import abstractmethod

import pennylane as qml
from pennylane.templates import AmplitudeEmbedding

from qcc.quantum import to_qubits, wires_to_qubits
from qcc.quantum.operation import Qubits, QubitsProperty
from qcc.ml import is_iterable, Module
from qcc.file import draw

if TYPE_CHECKING:
    from typing import Iterable, Optional
    from numbers import Number
    from pennylane.wires import Wires
    from pennylane.operation import Operation
    from qcc.quantum.operation import Parameters

    Statevector = Iterable[Number]


def is_multidimensional(wires: Qubits):
    if is_iterable(wires):
        return any(is_iterable(w) for w in wires)
    return False


class Ansatz(Module):
    __slots__ = "_qubits", "_num_layers"

    qubits: Qubits = QubitsProperty(slots=True)
    _num_layers: int

    def __init__(self, qubits, num_layers: int = 0):
        self.qubits = qubits
        self.num_layers = num_layers

    # Main properties

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
        wires = self.qubits.flatten()[::-1]  # Big-endian format
        device = qml.device("default.qubit", wires=wires)
        return qml.QNode(self._circuit, device, interface="torch")

    # Abstract methods

    @abstractmethod
    def circuit(self, *params: Parameters) -> Wires:
        pass

    @property
    @abstractmethod
    def max_layers(self) -> int:
        pass

    # Circuit operation

    @property
    def _data_wires(self) -> Wires:
        return self.qubits.flatten()

    def c2q(self, psi_in: Statevector) -> Operation:
        wires = self._data_wires[::-1]
        return AmplitudeEmbedding(psi_in, wires, pad_with=0, normalize=True)

    def q2c(self, wires: Wires):
        return qml.probs(wires)

    def post_processing(self, result) -> Iterable[Iterable[Number]]:
        # Makes sure batch is 2D array
        return result.unsqueeze(0) if result.dim() == 1 else result

    def _circuit(
        self,
        psi_in: Optional[Statevector] = None,
        params: Optional[Parameters] = None,
    ):
        if psi_in is not None:  # this is done to facilitate drawing
            self.c2q(psi_in)
        meas = self.circuit(*self.parameters() if params is None else params)
        return self.q2c(meas)

    def forward(self, psi_in: Optional[Statevector] = None):
        result = self.qnode(psi_in=psi_in)  # pylint: disable=not-callable

        return self.post_processing(result)

    # Miscellaneous

    def draw(self, filename=None, include_axis: bool = False, decompose: bool = False):
        expansion_strategy = "device" if decompose else "gradient"
        fig, ax = qml.draw_mpl(self.qnode, expansion_strategy=expansion_strategy)()

        return draw((fig, ax), filename, overwrite=False, include_axis=include_axis)

    # Instance factories

    @classmethod
    def from_dims(cls, dims: Iterable[int], *args, **kwargs):
        dims_q = to_qubits(dims)
        qubits = wires_to_qubits(dims_q)

        ansatz = cls(qubits, *args, **kwargs)

        return ansatz
