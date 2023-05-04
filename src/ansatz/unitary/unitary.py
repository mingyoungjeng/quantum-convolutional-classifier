from abc import ABC, abstractmethod
from typing import Sequence, Tuple
from numbers import Number

Parameters = Sequence[Number]
Wires = Sequence[int]


class Unitary(ABC):
    """
    A quantum unitary operation
    """

    @classmethod
    def __call__(cls, params: Parameters, wires: Wires) -> Parameters:
        """
        Performs a unitary operation

        Args:
            params (Parameters): parameters to unitary operation
            wires (Wires): target qubits

        Returns:
            Parameters: unused parameters
        """
        # Extract only required parameters
        active_params, params = cls.cut(params, cls.total_params(wires))

        # Execute unitary operation
        cls._u(active_params, wires)

        # Return leftover parameters
        return params

    @staticmethod
    @abstractmethod
    def _u(params: Parameters, wires: Wires) -> None:
        """
        Performs a unitary operation (implementation)

        Args:
            params (Parameters): parameters to unitary operation
            wires (Wires): target qubits
        """

    @staticmethod
    def cut(arr: Parameters, i: int) -> Tuple[Parameters, Parameters]:
        """
        Splits sequence at index i

        Args:
            arr (Sequence[Number]): Sequence to split
            i (int): index to divide

        Returns:
            Tuple[Sequence[Number], Sequence[Number]]: split sub-sequences
        """
        return arr[:i], arr[i:]

    @classmethod
    @abstractmethod
    def total_params(cls, wires: Wires) -> int:
        """
        Total parameters required when applying operation to a set of qubits

        Args:
            wires (Wires): target qubits

        Returns:
            int: required number of parameters
        """
