from abc import ABC, abstractmethod
from typing import Sequence

from ansatz.unitary.unitary import Parameters, Wires


class Ansatz(ABC):
    @abstractmethod
    def __call__(self, params: Parameters) -> Wires:
        pass

    @property
    @abstractmethod
    def total_params(self) -> int:
        """
        Minimum number of parameters required for execution

        Returns:
            int: number of parameters
        """


class ConvolutionAnsatz(Ansatz):
    @property
    @abstractmethod
    def max_layers(self) -> int:
        """
        Most number of layers supported by ansatz

        Returns:
            int: maximum layers
        """
