from __future__ import annotations

from typing import Iterable
from pennylane.wires import Wires


class Qubits(list):
    def __init__(self, iterable=None):
        if iterable is None:
            super().__init__()
        else:
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


class QubitsProperty:
    __slots__ = "name", "slots"

    def __init__(self, name: str = None, slots: bool = False) -> None:
        self.name = name
        self.slots = slots

    def __set_name__(self, owner, name) -> None:
        if self.name is not None:
            return

        self.name = f"_{name}" if self.slots else name

    def __get__(self, obj: object, type=None) -> Qubits:
        if self.slots:
            qubits = getattr(obj, self.name, Qubits())
        else:
            qubits = obj.__dict__.get(self.name, Qubits())
        return qubits.copy()

    def __set__(self, obj: object, value: Iterable) -> None:
        if self.slots:
            setattr(obj, self.name, Qubits(value))
        else:
            obj.__dict__[self.name] = Qubits(value)
