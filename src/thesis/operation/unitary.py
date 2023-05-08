from __future__ import annotations
from typing import TYPE_CHECKING

from attrs import define, field
from abc import ABC, abstractmethod
import numpy as np
import pennylane as qml
from pennylane.wires import Wires
from pennylane.operation import Operation, AnyWires
from thesis.fn.quantum import to_qubits
from thesis.fn.machine_learning import is_iterable

if TYPE_CHECKING:
    from typing import Iterable
    from numbers import Number

    Parameters = Iterable[Number]
    Qubits = int | Iterable[int | Iterable[int]]


class Unitary(Operation):
    """
    A quantum unitary operation
    """

    num_wires = AnyWires

    @property
    def num_params(self) -> int:
        return 1

    @staticmethod
    @abstractmethod
    def shape(wires: Wires) -> int:
        """
        Total trainable parameters required when applying operation to a set of qubits

        Args:
            wires (Wires): target qubits

        Returns:
            int: required number of parameters
        """


def is_multidimensional(wires: Wires):
    if is_iterable(wires):
        return any(is_iterable(w) for w in wires)
    return False


def _convert_qubits(q: Qubits) -> Qubits:
    return [Wires(w) for w in q] if is_multidimensional(q) else Wires(q)


@define(frozen=True)
class Ansatz(ABC):
    qubits: Qubits = field(converter=_convert_qubits)
    num_layers: int = field()

    @num_layers.default
    def _num_layers(self):
        return self.max_layers

    @num_layers.validator
    def _check_num_layers(self, _, value):
        if value < 0:
            raise ValueError(f"Number of layers {value} can't be less than 0")

        if value > self.max_layers:
            raise ValueError(
                f"Number of layers {value} exceeds maximum number of layers {self.max_layers}"
            )

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

    @abstractmethod
    def __call__(self, params: Parameters) -> Wires:
        pass

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
