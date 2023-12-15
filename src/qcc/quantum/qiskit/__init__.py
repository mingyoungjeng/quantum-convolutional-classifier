"""
Qiskit implementations of common quantum operations.

!! ASSUME EVERYTHING IN qcc.quantum.qiskit.ansatz HAS NOT BEEN MAINTAINED AND IS OUTDATED !!
"""

from ..qubits import TwoDimensionalList as Qubits
from .perfect_shuffle import PerfectShuffle, RotateLeft, RotateRight
from .shift import Shift, Incrementor, Decrementor, CShift
from .mac import MultiplyAndAccumulate
from .c2q import C2Q
from .convolution import Convolution
