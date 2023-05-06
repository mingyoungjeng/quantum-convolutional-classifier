from __future__ import annotations
from typing import TYPE_CHECKING

from abc import abstractmethod
from thesis.unitary import Unitary

if TYPE_CHECKING:
    from thesis.unitary import Parameters, Wires


class Ansatz(Unitary):
    @property
    @abstractmethod
    def max_layers(self, *args, **kwargs) -> int:
        """
        Most number of layers supported by ansatz

        Returns:
            int: maximum layers
        """
