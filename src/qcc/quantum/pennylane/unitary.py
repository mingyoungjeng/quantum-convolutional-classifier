from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from abc import abstractmethod
from numbers import Number
from pennylane.wires import Wires
from pennylane.operation import Operation, AnyWires

if TYPE_CHECKING:
    Parameters = Iterable[Number]


class Unitary(Operation):
    """Base class for a quantum unitary operation"""

    num_wires = AnyWires

    @property
    def num_params(self) -> int:
        return 1

    @staticmethod
    @abstractmethod
    def _shape(num_wires: int, **hyperparameters) -> int:
        pass

    @classmethod
    def shape(cls, wires: Wires | int | None = None, **hyperparameters) -> int:
        """
        Total trainable parameters required when applying operation to a set of qubits

        Args:
            wires (Wires | int): target qubits or number of qubits

        Returns:
            int: required number of parameters
        """
        if isinstance(wires, Wires):
            wires = len(wires)
        return cls._shape(wires, **hyperparameters)
