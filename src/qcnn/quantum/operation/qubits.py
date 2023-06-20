from __future__ import annotations

from typing import Iterable
from pennylane.wires import Wires


class Qubits(list):
    def __init__(self, iterable):
        super().__init__(self._convert(item) for item in iterable)

    def __setitem__(self, index, item):
        super().__setitem__(index, self._convert(item))

    def insert(self, index, item):
        super().insert(index, self._convert(item))

    def append(self, item):
        super().append(self._convert(item))

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(self._convert(item) for item in other)

    def __add__(self, value) -> Qubits:
        return Qubits(super().__add__(value))

    def __iadd__(self, value) -> Qubits:
        self.extend(value)
        return self

    def copy(self) -> Qubits:
        return Qubits(super().copy())

    @staticmethod
    def _convert(item):
        if not isinstance(item, Iterable):
            item = [item]
        return Wires.all_wires(item)

    def flatten(self) -> Wires:
        return Wires.all_wires(self)

    @property
    def total(self) -> int:
        return len(self.flatten())

    @property
    def shape(self) -> Iterable[int] | int:
        shape = [len(q) for q in self]
        return shape and shape[0]

    @property
    def ndim(self) -> int:
        return len(self)
