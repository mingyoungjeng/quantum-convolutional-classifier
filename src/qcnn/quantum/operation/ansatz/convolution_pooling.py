from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import chain, zip_longest
from torch.nn import Module
from pennylane.wires import Wires

from qcnn.ml.optimize import init_params
from qcnn.quantum import to_qubits
from qcnn.quantum.operation import Convolution, Multiplex, Qubits
from qcnn.quantum.operation.ansatz import Ansatz
from qcnn.quantum.operation.c2q import ConvolutionAngleFilter
from qcnn.quantum.operation.fully_connected import FullyConnected

if TYPE_CHECKING:
    from typing import Iterable
    from qcnn.quantum.operation import Parameters, Unitary


class ConvolutionPoolingAnsatz(Ansatz):
    __slots__ = (
        "_ancilla_qubits",
        "_feature_qubits",
        "U_filter",
        "U_fully_connected",
        "filter_shape",
        "num_features",
        "pre_op",
        "post_op",
    )

    _ancilla_qubits: Qubits
    _feature_qubits: Qubits

    U_filter: Unitary
    U_fully_connected: Unitary

    num_features: int
    pre_op: bool
    post_op: bool

    filter_shape: Iterable[int]

    def __init__(
        self,
        qubits,
        num_layers=None,
        U_filter=ConvolutionAngleFilter,
        U_fully_connected=FullyConnected,
        num_features=1,
        pre_op=False,
        post_op=False,
        filter_shape=(2, 2),
    ):
        Module.__init__(self)
        self.main_qubits = qubits
        self.U_filter = U_filter
        self.U_fully_connected = U_fully_connected
        self.num_features = num_features
        self.pre_op = pre_op
        self.post_op = post_op
        self.filter_shape = filter_shape

        # Setup feature and ancilla qubits
        self._setup_features()
        self._setup_ancilla()

        self.num_layers = num_layers
        self._params = init_params(self.shape, angle=True)

    def circuit(self, *params: Parameters) -> Wires:
        (params,) = params
        n_dim = self.main_qubits.ndim
        main_qubits = self.main_qubits
        main_qubits += [[] for _ in range(len(self.filter_shape))]

        if self.pre_op:  # Pre-op on ancillas
            params = self._filter(self.ancilla_qubits, params)

        # Convolution layers
        for i in range(self.num_layers):
            qubits = main_qubits + self.ancilla_qubits

            ### SHIFT
            Convolution.shift(self._filter_shape_qubits, qubits, H=False)

            ### FILTER
            params = self._filter(qubits, params)

            ### PERMUTE
            for j, fsq in enumerate(self._filter_shape_qubits):
                main_qubits[n_dim + j] += main_qubits[j][:fsq]
                main_qubits[j] = main_qubits[j][fsq:]

        if self.post_op:  # Post-op on ancillas
            params = self._filter(self.ancilla_qubits, params)

        # Fully connected layer
        meas = Qubits(
            chain(*zip_longest(self.ancilla_qubits, main_qubits[:n_dim], fillvalue=[]))
        )
        meas += main_qubits[n_dim:] + self.feature_qubits

        meas = meas.flatten()
        self.U_fully_connected(params, meas[::-1])

        return meas[-1]

    @property
    def shape(self) -> int:
        n_params = self._n_params * (self.num_layers + self.pre_op + self.post_op)
        n_params += self.U_fully_connected.shape(self.qubits.flatten())

        return n_params

    @property
    def max_layers(self) -> int:
        return min((len(q) for q in self.main_qubits))

    def c2q(self, psi_in, _=None):
        return super().c2q(psi_in=psi_in, wires=self._data_qubits.flatten())

    ### QUBIT PROPERTIES

    @property
    def qubits(self) -> Qubits:
        return self._data_qubits + self.feature_qubits

    @property
    def main_qubits(self) -> Qubits:
        return self._qubits.copy()

    @main_qubits.setter
    def main_qubits(self, q) -> None:
        self._qubits = Qubits(q)

    @property
    def feature_qubits(self) -> Qubits:
        return self._feature_qubits.copy()

    @feature_qubits.setter
    def feature_qubits(self, q):
        self._feature_qubits = Qubits(q)

    @property
    def ancilla_qubits(self) -> Qubits:
        return self._ancilla_qubits.copy()

    @ancilla_qubits.setter
    def ancilla_qubits(self, q) -> None:
        self._ancilla_qubits = Qubits(q)

    ### PRIVATE

    def _setup_features(self) -> None:
        top = self.main_qubits.total
        self.feature_qubits = Qubits([range(top, top + to_qubits(self.num_features))])
        top += to_qubits(self.num_features)

    def _setup_ancilla(self) -> None:
        self.ancilla_qubits = []
        qubits = self.main_qubits
        for i, fsq in enumerate(self._filter_shape_qubits):
            self.ancilla_qubits += [qubits[i][:fsq]]
            qubits[i] = qubits[i][fsq:]
        self.main_qubits = qubits

    def _filter(self, qubits, params):
        """Wrapper around self.U_filter that replaces Convolution.filter"""

        # Setup params
        filter_params, params = params[: self._n_params], params[self._n_params :]
        shape = (self.num_features, len(filter_params) // self.num_features)
        filter_params = filter_params.reshape(shape)

        # Setup wires
        qubits = Qubits(q[:fsq] for q, fsq in zip(qubits, self._filter_shape_qubits))
        Multiplex(
            filter_params,
            qubits.flatten(),
            self.feature_qubits.flatten(),
            self.U_filter,
        )

        return params  # Leftover parameters

    @property
    def _data_qubits(self) -> Qubits:
        data_qubits = zip_longest(self.ancilla_qubits, self.main_qubits, fillvalue=[])
        return Qubits(chain(*data_qubits))

    @property
    def _filter_shape_qubits(self):
        return to_qubits(self.filter_shape)

    @property
    def _n_params(self) -> int:
        return self.num_features * self.U_filter.shape(sum(self._filter_shape_qubits))
