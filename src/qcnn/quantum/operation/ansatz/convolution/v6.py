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
from qcnn.quantum.operation.ansatz.basic import (
    BasicFiltering,
    BasicFiltering2,
    BasicFiltering3,
    BasicFiltering4,
)
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
    U_filter = BasicFiltering
    U_fully_connected = BasicFiltering4

    def __init__(self, qubits, num_layers=None, num_features=1, measure_all=True):
        Module.__init__(self)
        self.main_qubits = qubits
        self._feature_qubits = []
        self._ancilla_qubits = []
        self.num_layers = num_layers
        self.num_features = num_features
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

    def _filter(self, qubits, params):
        """Wrapper around self.U_filter that replaces Convolution.filter"""

        # Setup params
        filter_params, params = params[: self.n_params], params[self.n_params :]
        shape = (self.num_features, len(filter_params) // self.num_features)
        filter_params = filter_params.reshape(shape)

        # Setup wires
        wires = [q[:fsq] for q, fsq in zip(qubits, self.filter_shape_qubits)]
        wires = Wires.all_wires(wires)

        Multiplex(filter_params, wires, self.feature_wires, self.U_filter)

        return params  # Leftover parameters

    def circuit(self, params: Parameters) -> Wires:
        n_dim = len(self.main_qubits)
        main_qubits = self.main_qubits
        main_qubits += [[] for _ in range(len(self.filter_shape))]

        # Pre-op on ancillas
        params = self._filter(
            [main_qubits[i][:fsq] for i, fsq in enumerate(self.filter_shape_qubits)],
            params,
        )

        # Convolution layers
        for i in range(self.num_layers):
            # Setup ancilla qubits
            for i, fsq in enumerate(self.filter_shape_qubits):
                self.ancilla_qubits += [main_qubits[i][:fsq]]
                main_qubits[i] = main_qubits[i][fsq:]

            qubits = main_qubits + self.ancilla_qubits

            Convolution.shift(self.filter_shape_qubits, qubits, H=False)

            ### FILTER
            params = self._filter(qubits, params)

            ### PERMUTE
            # Convolution.permute(self.filter_shape_qubits, qubits)
            # for j, fsq in enumerate(self.filter_shape_qubits):
            #     main_qubits[n_dim + j] += main_qubits[j][:fsq]
            #     main_qubits[j] = main_qubits[j][fsq:]

        # Post-op on ancillas
        # params = self._filter(self.ancilla_qubits, params)

        # Fully connected layer
        meas = (
            list(chain(*zip(self.ancilla_qubits, main_qubits[:n_dim])))
            if self.measure_all
            else main_qubits[:n_dim]
        )
        meas += main_qubits[n_dim:] + self.feature_qubits

        meas = Wires.all_wires(meas)
        self.U_fully_connected(params, meas[::-1])

        return meas[-1]

    @property
    def shape(self) -> int:
        n_params = (self.num_layers + 1) * self.n_params

        n_meas = self.num_wires - (
            (self.num_layers - 1) * sum(self.filter_shape_qubits)
        )
        n_params += self.U_fully_connected.shape(range(n_meas))

        return n_params

    @property
    def max_layers(self) -> int:
        return self.min_dim - 1

    @property
    def n_params(self) -> int:
        wires = range(sum(self.filter_shape_qubits))
        return self.num_features * self.U_filter.shape(wires)
