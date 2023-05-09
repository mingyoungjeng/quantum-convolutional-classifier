from __future__ import annotations
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
from attrs import define, field, validators
import numpy as np
import pennylane as qml

from pennylane.wires import Wires
from pennylane.templates import AmplitudeEmbedding
from thesis.quantum import to_qubits
from thesis.ml.ml import is_iterable

if TYPE_CHECKING:
    from typing import Iterable
    from numbers import Number
    from pennylane.operation import Operation
    from thesis.operation import Parameters, Qubits


def is_multidimensional(wires: Qubits):
    if is_iterable(wires):
        return any(is_iterable(w) for w in wires)
    return False


def _convert_qubits(q: Qubits) -> Qubits:
    return [Wires(w) for w in q] if is_multidimensional(q) else Wires(q)


def _check_num_layers(self: Ansatz, _, value):
    if value > self.max_layers:
        raise ValueError(
            f"Number of layers {value} exceeds maximum number of layers {self.max_layers}"
        )


@define(frozen=True)
class Ansatz(ABC):
    qubits: Qubits = field(converter=_convert_qubits)
    _qnode: qml.QNode = field(init=False, repr=False)

    @_qnode.default  # @qubits.validator
    def _check_qnode(self):
        device = qml.device("default.qubit", wires=self.num_wires)
        return qml.QNode(self.__call__, device, interface="torch")

    num_layers: int = field(validator=[validators.ge(0), _check_num_layers])

    @num_layers.default
    def _num_layers(self):
        return self.max_layers

    @property
    def wires(self) -> Wires:
        return Wires.all_wires(self.qubits)

    @property
    def num_wires(self) -> int:
        return len(self.wires)

    @property
    def num_qubits(self) -> Iterable[int] | int:
        if is_multidimensional(self.qubits):
            return [len(w) for w in self.qubits]
        else:
            return self.num_wires

    def c2q(self, psi_in: Iterable[Number]) -> Operation:
        return AmplitudeEmbedding(psi_in, self.wires, pad_with=0, normalize=True)

    @abstractmethod
    def circuit(self, params: Parameters) -> Wires:
        pass

    def q2c(self, wires: Wires):
        return qml.probs(wires)

    def __call__(self, params: Parameters, psi_in: Iterable[Number]) -> Iterable[float]:
        self.c2q(psi_in)
        meas = self.circuit(params)
        return self.q2c(meas)

    @property
    @abstractmethod
    def shape(self) -> int:
        pass

    @property
    @abstractmethod
    def max_layers(self) -> int:
        pass

    @classmethod
    def from_dims(cls, dims: Iterable[int], *args, **kwargs):
        dims = to_qubits(dims)
        qubits = [list(range(x - y, x)) for x, y in zip(np.cumsum(dims), dims)]
        return cls(qubits, *args, **kwargs)

    @property
    def qnode(self):
        return self._qnode


# class Ansatz(Unitary):
#     def __init__(self, *params, wires=None, do_queue=True, id=None, **kwargs):
#         self._hyperparameters = kwargs

#         if is_multidimensional(wires):
#             wires = [qml.wires.Wires(w) for w in wires]
#             super().__init__(*params, wires=sum(wires), do_queue=do_queue, id=id)
#             self._wires = wires
#         else:
#             super().__init__(*params, wires=wires, do_queue=do_queue, id=id)

#     @property
#     def num_wires(self):
#         if is_multidimensional(self.wires):
#             return sum(len(w) for w in self.wires)
#         return super().num_wires

#     @property
#     @abstractmethod
#     def max_layers(self, *args, **kwargs) -> int:
#         """
#         Most number of layers supported by ansatz

#         Returns:
#             int: maximum layers
#         """
