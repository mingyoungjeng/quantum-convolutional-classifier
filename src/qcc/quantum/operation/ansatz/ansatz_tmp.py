from __future__ import annotations
from typing import TYPE_CHECKING

import logging
from abc import abstractmethod, ABCMeta

import pennylane as qml
from pennylane.qnn import TorchLayer
from pennylane.templates import AmplitudeEmbedding

from qcc.quantum import to_qubits, wires_to_qubits
from qcc.quantum.operation import Qubits, QubitsProperty
from qcc.ml import reset_parameter
from qcc.file import draw

if TYPE_CHECKING:
    from typing import Iterable, Optional
    from numbers import Number
    from pennylane.wires import Wires
    from pennylane.operation import Operation
    from qcc.quantum.operation import Parameters

    Statevector = Iterable[Number]

log = logging.getLogger(__name__)


class Ansatz(TorchLayer, metaclass=ABCMeta):
    __slots__ = "_qubits", "_num_layers"

    qubits: Qubits = QubitsProperty(slots=True)
    _num_layers: int

    def __init__(self, qubits: Qubits, num_layers: int = 0):
        self.qubits = qubits
        self.num_layers = num_layers

        wires = self.qubits.flatten()[::-1]  # Big-endian format
        device = qml.device("default.qubit", wires=wires)
        qnode = qml.QNode(self._circuit, device, interface="torch")

        super().__init__(qnode, {"params": self.shape})

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

    # Abstract methods

    @property
    @abstractmethod
    def shape(self) -> int:
        pass

    @abstractmethod
    def circuit(self, *params: Parameters) -> Wires:
        pass

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
        return qml.probs(wires)

    def _circuit(
        self,
        params: Parameters,
        inputs: Optional[Statevector] = None,
    ):
        if inputs is not None:  # this is done to facilitate drawing
            self.c2q(inputs)
        meas = self.circuit(params)
        return self.q2c(meas)

    # Miscellaneous

    def reset_parameters(self):
        for parameter in self.parameters():
            reset_parameter(parameter)

    def draw(self, filename=None, include_axis: bool = False, decompose: bool = False):
        expansion_strategy = "device" if decompose else "gradient"
        fig, ax = qml.draw_mpl(self.qnode, expansion_strategy=expansion_strategy)(
            **self.qnode_weights
        )

        return draw((fig, ax), filename, overwrite=False, include_axis=include_axis)

    # Pennylane why did you do this

    def __getattr__(self, item):
        return super(TorchLayer, self).__getattr__(item)

    def __setattr__(self, item, val):
        super(TorchLayer, self).__setattr__(item, val)

    # Instance factories

    @classmethod
    def from_dims(cls, dims: Iterable[int], *args, **kwargs):
        dims_q = to_qubits(dims)
        qubits = wires_to_qubits(dims_q)

        self = cls(qubits, *args, **kwargs)

        info = qml.specs(self.qnode, expansion_strategy="device")
        info = info(**self.qnode_weights)["resources"]
        log.info(f"Depth: {info.depth}")
        gate_count = sum(key * value for key, value in info.gate_sizes.items())
        log.info(f"Gate Count: {gate_count}")

        return self
