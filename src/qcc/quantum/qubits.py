"""
Qubits
Representation of qubits for multidimensional data as a 2D list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable
from pennylane.wires import Wires

if TYPE_CHECKING:
    from typing import Any, Callable


class TwoDimensionalList(list):
    """List of lists"""

    __slots__ = "fn"

    # ==== Overrides of list ==== #

    def __init__(self, iterable=None, fn: Callable[[Any], Any] = None):
        self.fn = lambda x: x if fn is None else fn
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

    def __add__(self, value) -> TwoDimensionalList:
        return TwoDimensionalList(super().__add__(value))

    def __iadd__(self, value) -> TwoDimensionalList:
        self.extend(value)
        return self

    def copy(self) -> TwoDimensionalList:
        return TwoDimensionalList(super().copy())

    # ==== Extensions to list ==== #

    def _convert(self, item):
        if not isinstance(item, Iterable):
            item = [item]
        return [self.fn(i) for i in item]

    def flatten(self) -> list:
        return [item for dim in self for item in dim]

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


class QubitsPennylane(TwoDimensionalList):
    def __init__(self, iterable=None):
        super().__init__(iterable, fn=Wires.all_wires)

    def _convert(self, item):
        if not isinstance(item, Iterable):
            item = [item]
        return Wires.all_wires(item)

    def flatten(self) -> Wires:
        return Wires.all_wires(self)


class QubitsPennylaneProperty:
    """Descriptor that handles getting and setting Qubits in classes"""

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

    def __get__(self, obj: object, _=None) -> QubitsPennylane:
        if self.use_slots:
            qubits = getattr(obj, self.name, QubitsPennylane())
        else:
            qubits = obj.__dict__.get(self.name, QubitsPennylane())
        return qubits.copy()

    def __set__(self, obj: object, value: Iterable) -> None:
        if self.use_slots:
            setattr(obj, self.name, QubitsPennylane(value))
        else:
            obj.__dict__[self.name] = QubitsPennylane(value)
