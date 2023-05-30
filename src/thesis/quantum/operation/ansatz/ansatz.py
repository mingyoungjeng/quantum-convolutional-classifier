from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
from attrs import define, field, validators, Factory

import pennylane as qml
from pennylane.wires import Wires
from pennylane.templates import AmplitudeEmbedding

from thesis.quantum import to_qubits, wires_to_qubits
from thesis.ml import is_iterable
from thesis.ml.optimize import init_params

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
        validator=[
            validators.ge(0),
            lambda *x: validators.le(x[0].max_layers)(*x),
        ],
        default=Factory(lambda self: self.max_layers, takes_self=True),
    )
    parameters = field(
        init=False,
        repr=False,
        default=Factory(lambda self: init_params(self.shape), takes_self=True),
    )
    qnode: qml.QNode = field(init=False, repr=False)

    @qnode.default  # @qubits.validator
    def _check_qnode(self):
        device = qml.device("default.qubit", wires=self.wires[::-1])
        return qml.QNode(self._circuit, device, interface="torch")

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

    def post_processing(self, result):
        # Makes sure batch is 2D array
        if result.dim() == 1:
            result = result.unsqueeze(0)

        return result

    def _circuit(
        self,
        params: Optional[Parameters] = None,
        psi_in: Optional[Statevector] = None,
    ):
        if psi_in is not None:  # this is done to facilitate drawing
            self.c2q(psi_in)
        meas = self.circuit(self.parameters if params is None else params)
        return self.q2c(meas)

    def __call__(self, psi_in: Optional[Statevector] = None):
        result = self.qnode(psi_in=psi_in)  # pylint: disable=not-callable

        return self.post_processing(result)

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
