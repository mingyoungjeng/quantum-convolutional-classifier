from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

import logging
from enum import StrEnum
from abc import abstractmethod, ABCMeta

import torch
from torch.nn import Module
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.templates import AmplitudeEmbedding

from qcc.quantum import to_qubits, wires_to_qubits
from qcc.quantum.pennylane import Qubits, QubitsProperty
from qcc.ml import init_params, reset_parameter
from qcc.file import draw

if TYPE_CHECKING:
    from numbers import Number
    from pennylane.wires import Wires
    from pennylane.operation import Operation
    from qcc.quantum.pennylane import Parameters
    from torch import Tensor

    Statevector = Iterable[Number]

log = logging.getLogger(__name__)


class Ansatz(Module, metaclass=ABCMeta):
    """Base class for QML ansatz"""

    __slots__ = "_qubits", "_num_layers", "_qnode", "num_classes", "q2c_method"

    qubits: Qubits = QubitsProperty(slots=True)
    _num_layers: int
    num_classes: int | None
    q2c_method: Q2CMethod

    class Q2CMethod(StrEnum):
        Probabilities = "probs"
        ExpectationValue = "expval"
        Parity = "parity"

    def __init__(
        self,
        qubits: Qubits,
        num_layers: int = 0,
        num_classes: int | None = None,
        q2c_method: Q2CMethod | str = Q2CMethod.Probabilities,
    ):
        super().__init__()
        self.qubits = qubits
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.q2c_method = self.Q2CMethod(q2c_method)

        self.register_parameter("weight", init_params(self.shape))
        self.reset_parameters()

        wires = self.qubits.flatten()[::-1]  # Big-endian format
        device = qml.device("default.qubit", wires=wires)
        self._qnode = qml.QNode(self._circuit, device, interface="torch")

    # Main properties

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value: int) -> None:
        # Check bounds
        if value < 0 or value > self.max_layers:
            err = f"num_layers must be in range (0, {self.max_layers}], got {value}"
            raise ValueError(err)

        self._num_layers = value

    # Derived properties

    @property
    def qnode(self) -> qml.QNode:
        return self._qnode

    @qnode.deleter
    def qnode(self) -> None:
        del self._qnode

    # Abstract methods

    @property
    @abstractmethod
    def shape(self) -> int:
        pass

    @abstractmethod
    def circuit(self, *params: Parameters) -> Wires:
        """
        Quantum circuit definition of ansatz

        Returns:
            Wires: Qubits to measure in little endian order (like in Qiskit)
        """

    @property
    @abstractmethod
    def max_layers(self) -> int:
        pass

    # Circuit operation

    @property
    def _data_wires(self) -> Wires:
        return self.qubits.flatten()

    def c2q(self, psi_in: Statevector) -> Operation:
        wires = self._data_wires[::-1]
        return AmplitudeEmbedding(psi_in, wires, pad_with=0, normalize=True)

    def q2c(self, wires: Wires):
        # Converts wires from little-endian to big-endian
        wires = wires[::-1] if isinstance(wires, Iterable) else wires

        match self.q2c_method:
            case self.Q2CMethod.Probabilities:
                return qml.probs(wires)
            case self.Q2CMethod.ExpectationValue:
                return tuple(qml.expval(qml.PauliZ(w)) for w in wires)
            case self.Q2CMethod.Parity:
                return qml.expval(Tensor(*(qml.PauliZ(w) for w in wires)))
            case _:
                return

    def _circuit(
        self,
        psi_in: Statevector | None = None,
        params: Parameters | None = None,
    ):
        if psi_in is not None:  # this is done to facilitate drawing
            self.c2q(psi_in)
        meas = self.circuit(*self.parameters() if params is None else params)
        return self.q2c(meas)

    def forward(self, psi_in: Statevector | None = None) -> Tensor:
        result = self.qnode(psi_in=psi_in)  # pylint: disable=not-callable

        match self.q2c_method:
            case self.Q2CMethod.ExpectationValue:
                result = torch.vstack(result).T
                result = (result + 1) / 2
            # case self.Q2CMethod.Probabilities:
            #     result = torch.sqrt(result)
            case self.Q2CMethod.Parity:
                result = (result + 1) / 2
                result = torch.vstack((result, 1 - result)).T
            case _:
                pass

        # Makes sure batch is 2D array
        if result.dim() == 1:
            result = result.unsqueeze(0)

        result = self._forward(result)

        return result

    def _forward(self, result):
        return result

    # Miscellaneous

    @property
    def _num_meas(self) -> int:
        if self.num_classes is None:
            return self.qubits.total

        match self.q2c_method:
            case self.Q2CMethod.Probabilities:
                return to_qubits(self.num_classes)
            case self.Q2CMethod.ExpectationValue:
                return self.num_classes
            case self.Q2CMethod.Parity:
                return self.qubits.total
            case _:
                return 0

    def reset_parameters(self):
        reset_parameter(self.get_parameter("weight"))

    def draw(self, filename=None, include_axis: bool = False, decompose: bool = False):
        expansion_strategy = "device" if decompose else "gradient"
        fig, ax = qml.draw_mpl(self.qnode, expansion_strategy=expansion_strategy)()

        return draw((fig, ax), filename, overwrite=False, include_axis=include_axis)

    # Instance factories

    @classmethod
    def from_dims(cls, dims: int | Iterable[int], *args, **kwargs):
        if isinstance(dims, int):
            dims = [dims]

        dims_q = to_qubits(dims)
        qubits = wires_to_qubits(dims_q)

        self = cls(qubits, *args, **kwargs)

        info = qml.specs(self.qnode, expansion_strategy="device")()["resources"]
        log.info(f"Depth: {info.depth}")
        gate_count = sum(key * value for key, value in info.gate_sizes.items())
        log.info(f"Gate Count: {gate_count}")

        return self
