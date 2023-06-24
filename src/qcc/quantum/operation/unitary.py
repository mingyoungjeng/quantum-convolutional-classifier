from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from abc import abstractmethod
from numbers import Number
from pennylane.wires import Wires
from pennylane.operation import Operation, AnyWires

if TYPE_CHECKING:
    from typing import Optional

    Parameters = Iterable[Number]


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
    def _shape(wires: Wires) -> int:
        pass

    @classmethod
    def shape(cls, wires: Optional[Wires | int] = None) -> int:
        """
        Total trainable parameters required when applying operation to a set of qubits

        Args:
            wires (Wires | int): target qubits or number of qubits

        Returns:
            int: required number of parameters
        """
        if isinstance(wires, Number):
            wires = range(wires)
        return cls._shape(wires)
