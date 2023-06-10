from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import chain
from torch.nn import Module
from pennylane.wires import Wires


from qcnn.ml.optimize import init_params
from qcnn.quantum import to_qubits
from qcnn.quantum.operation import Convolution, Multiplex
from qcnn.quantum.operation.ansatz import is_multidimensional
from qcnn.quantum.operation.ansatz.convolution.test import ConvolutionAnsatz as Base
from qcnn.quantum.operation.ansatz.basic import BasicFiltering, BasicFiltering2
from qcnn.quantum.operation.ansatz.simple import SimpleFiltering, SimpleFiltering2
from qcnn.quantum.operation.c2q import (
    ConvolutionAngleFilter,
    ConvolutionComplexAngleFilter,
)

if TYPE_CHECKING:
    from typing import Iterable
    from qcnn.quantum.operation import Parameters


class ConvolutionAnsatz(Base):
    __slots__ = "_feature_qubits"
    U_filter = ConvolutionComplexAngleFilter
    U_fully_connected = BasicFiltering
    num_features = 1

    def __init__(self, qubits, num_layers=None, measure_all=True):
        Module.__init__(self)
        self.main_qubits = qubits
        self._feature_qubits = []
        self._ancilla_qubits = []
        self.num_layers = num_layers
        self.measure_all = measure_all

        # Feature qubits
        top = self.num_wires
        self.feature_qubits = [list(range(top, top + to_qubits(self.num_features)))]
        top += to_qubits(self.num_features)

        self._params = init_params(self.shape)

    @property
    def qubits(self) -> Iterable[Iterable[int]]:
        return self.main_qubits + self.feature_qubits + self.ancilla_qubits

    @property
    def feature_qubits(self):
        return self._feature_qubits

    @feature_qubits.setter
    def feature_qubits(self, q):
        self._feature_qubits = (
            [Wires(w) for w in q] if is_multidimensional(q) else [Wires(q)]
        )

    @property
    def feature_wires(self):
        return Wires.all_wires(self.feature_qubits)

    def circuit(self, params: Parameters) -> Wires:
        n_dim = len(self.main_qubits)
        main_qubits = self.main_qubits
        main_qubits += [[] for _ in range(len(self.filter_shape))]

        # Setup ancilla qubits
        for i, fsq in enumerate(self.filter_shape_qubits):
            self.ancilla_qubits += [main_qubits[i][:fsq]]
            main_qubits[i] = main_qubits[i][fsq:]

        # Convolution layers
        for i in range(self.num_layers):
            conv_params, params = params[: self.n_params], params[self.n_params :]
            # conv_params = conv_params.reshape(self.filter_shape)

            qubits = main_qubits + self.ancilla_qubits

            Convolution.shift(self.filter_shape_qubits, qubits, H=False)

            ### FILTER
            wires = [q[:fsq] for q, fsq in zip(qubits, self.filter_shape_qubits)]
            wires = Wires.all_wires(wires)

            shape = (self.num_features, len(conv_params) // self.num_features)
            conv_params = conv_params.reshape(shape)

            Multiplex(conv_params, wires, self.feature_wires, self.U_filter)

            ### PERMUTE
            # Convolution.permute(self.filter_shape_qubits, qubits)
            for j, fsq in enumerate(self.filter_shape_qubits):
                main_qubits[n_dim + j] += main_qubits[j][:fsq]
                main_qubits[j] = main_qubits[j][fsq:]

        # Fully connected layer
        meas = (
            list(chain(*zip(self.ancilla_qubits, main_qubits[:n_dim])))
            if self.measure_all
            else main_qubits[:n_dim]
        )
        meas += main_qubits[n_dim:] + self.feature_qubits

        meas = Wires.all_wires(meas)
        self.U_fully_connected(params, meas)

        return meas[::-1]

    @property
    def shape(self) -> int:
        n_params = self.num_layers * self.n_params

        if self.measure_all:
            n_params += self.U_fully_connected.shape(self.wires)
        else:
            n_meas = self.num_wires - sum(self.filter_shape_qubits)
            n_params += self.U_fully_connected.shape(range(n_meas))

        return n_params

    @property
    def max_layers(self) -> int:
        return self.min_dim - 1

    @property
    def n_params(self) -> int:
        wires = range(sum(self.filter_shape_qubits))
        return self.num_features * self.U_filter.shape(wires)
