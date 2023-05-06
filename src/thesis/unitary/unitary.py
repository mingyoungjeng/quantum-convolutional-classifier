from __future__ import annotations
from typing import TYPE_CHECKING

from abc import abstractmethod
from pennylane.operation import Operation, AnyWires

if TYPE_CHECKING:
    from thesis.unitary import Wires


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
