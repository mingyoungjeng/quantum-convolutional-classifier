from __future__ import annotations
from typing import TYPE_CHECKING

from qcc.quantum.qubits import QubitsPennylane as Qubits
from qcc.quantum.qubits import QubitsPennylaneProperty as QubitsProperty

# Base class
from .unitary import Unitary

from .shift import Shift
from .c2q import C2Q
from .convolution import Convolution

if TYPE_CHECKING:
    from .unitary import Parameters
