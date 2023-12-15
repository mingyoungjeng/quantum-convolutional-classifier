"""
Qiskit implementations of common quantum operations.

!! ASSUME EVERYTHING IN qcc.quantum.qiskit.ansatz HAS NOT BEEN MAINTAINED AND IS OUTDATED !!
"""

from .c2q import C2Q
from .convolution import Convolution
from .mac import MultiplyAndAccumulate
from .perfect_shuffle import PerfectShuffle, RotateLeft, RotateRight
from .shift import Shift, Incrementor, Decrementor
from ..qubits import TwoDimensionalList as Qubits
