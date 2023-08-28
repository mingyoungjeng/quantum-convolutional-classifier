from __future__ import annotations
from typing import TYPE_CHECKING

import logging
from abc import abstractmethod, ABCMeta
from functools import partial

from torch.nn import Module
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

from qcc.quantum import to_qubits, wires_to_qubits
from qcc.quantum.pennylane import Qubits, QubitsProperty
from qcc.ml import init_params, reset_parameter
from qcc.file import draw
from qcc.quantum.qiskit.c2q import C2QAnsatz
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

if TYPE_CHECKING:
    from typing import Iterable, Optional
    from numbers import Number
    from pathlib import Path

    Statevector = Iterable[Number]

log = logging.getLogger(__name__)


def parity(x: int, /, num_classes: int = 2):
    return f"{x:b}".count("1") % num_classes


def q2c(x: int, /, meas, num_classes: int = 2):
    x = sum(((x >> b) & 1) << i for i, b in enumerate(sorted(meas)))
    return parity(x, num_classes)


class Ansatz(Module, metaclass=ABCMeta):
    __slots__ = "_qubits", "_num_layers", "num_classes", "module"

    qubits: Qubits = QubitsProperty(slots=True)
    _num_layers: int

    def __init__(self, qubits: Qubits, num_layers: int = 0, num_classes: int = 2):
        super().__init__()
        self.qubits = qubits
        self.num_layers = num_layers
        self.num_classes = num_classes

        qc, meas = self.circuit()
        weight_params = qc.parameters

        # C2Q
        c2q = C2QAnsatz(len(self.qubits.flatten()))
        qc.compose(c2q, qubits=self.qubits.flatten(), front=True, inplace=True)

        # Q2C
        # observables = "".join(
        #     "Z" if i in meas else "I" for i in range(len(self.qubits.flatten()))
        # )[::-1]
        # observables = SparsePauliOp([observables])
        if meas is None:
            interpret = partial(parity, num_classes=self.num_classes)
        else:
            interpret = partial(q2c, meas=meas, num_classes=self.num_classes)

        # Construct module

        # qnn = EstimatorQNN(
        #     circuit=qc,
        #     input_params=c2q.params,
        #     weight_params=weight_params,
        #     observables=observables,
        #     input_gradients=True,
        # )
        qnn = SamplerQNN(
            circuit=qc,
            input_params=c2q.params,
            weight_params=weight_params,
            interpret=interpret,
            output_shape=self.num_classes,
        )

        self.module = TorchConnector(qnn, initial_weights=init_params(qnn.num_weights))

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

    # Abstract methods

    @abstractmethod
    def circuit(self) -> tuple[QuantumCircuit, Optional[set[int]]]:
        pass

    @property
    @abstractmethod
    def max_layers(self) -> int:
        pass

    # Circuit operation

    def forward(self, psi_in: Optional[Statevector] = None):
        result = self.module.forward(psi_in)

        # Makes sure batch is 2D array
        return result.unsqueeze(0) if result.dim() == 1 else result

    # Miscellaneous

    def reset_parameters(self):
        for parameter in self.parameters():
            reset_parameter(parameter)

    def draw(self, filename: Optional[Path] = None, **_):
        fig = self.circuit()[0].draw("mpl", reverse_bits=True)
        return draw((fig, None), filename, overwrite=False, include_axis=False)

    # Instance factories

    @classmethod
    def from_dims(cls, dims: Iterable[int], *args, **kwargs):
        dims_q = to_qubits(dims)
        qubits = wires_to_qubits(dims_q)

        self = cls(qubits, *args, **kwargs)

        # TODO: depth and gate count

        return self
