from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
from attrs import define, field, validators
import numpy as np
import pennylane as qml

from pennylane.wires import Wires
from pennylane.templates import AmplitudeEmbedding
from thesis.quantum import to_qubits, wires_to_qubits
from thesis.ml.ml import is_iterable

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


@define(frozen=True)
class Ansatz(ABC):
    qubits: Qubits = field(
        converter=lambda q: [Wires(w) for w in q]
        if is_multidimensional(q)
        else [Wires(q)]
    )
    num_layers: int = field(
        validator=[validators.ge(0), lambda *x: validators.le(x[0].max_layers)(*x)]
    )
    _qnode: qml.QNode = field(init=False, repr=False)

    @num_layers.default
    def _num_layers(self):
        return self.max_layers

    @_qnode.default  # @qubits.validator
    def _check_qnode(self):
        device = qml.device("default.qubit", wires=self.wires[::-1])
        return qml.QNode(self.__call__, device, interface="torch")

    @property
    def qnode(self):
        return self._qnode

    # Derived properties

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
        return AmplitudeEmbedding(psi_in, self.wires, pad_with=0, normalize=True)

    def q2c(self, wires: Wires):
        return qml.probs(wires)

    def __call__(self, params: Parameters, psi_in: Optional[Statevector] = None):
        if psi_in is not None:
            self.c2q(psi_in)
        meas = self.circuit(params)
        return self.q2c(meas)

    # Abstract methods

    @abstractmethod
    def circuit(self, params: Parameters) -> Wires:
        pass

    @property
    @abstractmethod
    def shape(self) -> int:
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

        return cls(qubits, *args, **kwargs)
