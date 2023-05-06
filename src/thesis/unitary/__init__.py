from __future__ import annotations
from typing import TYPE_CHECKING

# Base class
from .unitary import Unitary

# Shift operation
from .shift import Shift

# Unitary operations for ML
from .baseline import (
    BaselineConvolution,
    BaselinePooling1,
    BaselinePooling2,
    BaselinePooling3,
)
from .simple import SimpleConvolution, SimplePooling

if TYPE_CHECKING:
    from typing import Sequence
    from numbers import Number

    Parameters = Sequence[Number]
    Wires = Sequence[int]
