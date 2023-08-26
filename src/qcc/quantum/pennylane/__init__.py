from __future__ import annotations
from typing import TYPE_CHECKING

# Qubits class
from .qubits import Qubits, QubitsProperty

# Base class
from .unitary import Unitary

from .shift import Shift
from .multiplex import Multiplex
from .c2q import C2Q
from .convolution import Convolution

if TYPE_CHECKING:
    from .unitary import Parameters
