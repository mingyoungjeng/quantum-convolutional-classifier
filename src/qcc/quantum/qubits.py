from __future__ import annotations

from typing import Iterable

from qiskit.circuit import QuantumRegister, Qubit
from pennylane.wires import Wires


class Qubits(list):
    """Multidimensional qubits"""

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

        # qreg = QuantumRegister(bits=[Qubit() for _ in item])
        return Wires.all_wires(item)
        # return item

    def flatten(self) -> Iterable:
        return [qubit for qregs in self for qubit in qregs]

    @property
    def total(self) -> int:
        return len(self.flatten())

    @property
    def shape(self) -> Iterable[int] | int:
        shape = tuple(len(q) for q in self)
        return shape[0] if len(shape) == 1 else shape

    @property
    def ndim(self) -> int:
        return len(self)


class QubitsProperty:
    """Descriptor that handles getting and setting Qubits"""

    __slots__ = "name", "use_slots"

    name: str
    use_slots: bool

    def __init__(self, name: str = None, slots: bool = False) -> None:
        self.name = name
        self.use_slots = slots

    def __set_name__(self, _, name) -> None:
        if self.name is not None:
            return

        self.name = f"_{name}" if self.use_slots else name

    def __get__(self, obj: object, _=None) -> Qubits:
        if self.use_slots:
            qubits = getattr(obj, self.name, Qubits())
        else:
            qubits = obj.__dict__.get(self.name, Qubits())
        return qubits.copy()

    def __set__(self, obj: object, value: Iterable) -> None:
        if self.use_slots:
            setattr(obj, self.name, Qubits(value))
        else:
            obj.__dict__[self.name] = Qubits(value)
