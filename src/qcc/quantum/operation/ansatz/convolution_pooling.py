from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import chain, zip_longest

from qcc.quantum import to_qubits
from qcc.quantum.operation import Convolution, Multiplex, Qubits, QubitsProperty
from qcc.quantum.operation.ansatz import Ansatz
from qcc.quantum.operation.c2q import ConvolutionAngleFilter
from qcc.quantum.operation.fully_connected import FullyConnected

if TYPE_CHECKING:
    from typing import Iterable
    from pennylane.wires import Wires
    from pennylane.operation import Operation
    from qcc.quantum.operation import Parameters, Unitary


class ConvolutionPoolingAnsatz(Ansatz):
    __slots__ = (
        "U_filter",
        "U_fully_connected",
        "filter_shape",
        "num_features",
        "pre_op",
        "post_op",
    )

    data_qubits: Qubits = QubitsProperty()
    ancilla_qubits: Qubits = QubitsProperty()
    feature_qubits: Qubits = QubitsProperty()

    filter_shape: Iterable[int]
    num_features: int

    U_filter: type[Unitary]
    U_fully_connected: type[Unitary]

    pre_op: bool
    post_op: bool

    def __init__(
        self,
        qubits: Qubits,
        num_layers: int = None,
        num_features: int = 1,
        filter_shape: Iterable[int] = (2, 2),
        U_filter: type[Unitary] = ConvolutionAngleFilter,
        U_fully_connected: type[Unitary] = FullyConnected,
        pre_op: bool = False,
        post_op: bool = False,
    ):
        self.data_qubits = qubits
        self._num_layers = num_layers
        self.num_features = num_features
        self.filter_shape = filter_shape

        self.U_filter = U_filter  # pylint: disable=invalid-name
        self.U_fully_connected = U_fully_connected  # pylint: disable=invalid-name
        self.pre_op = pre_op
        self.post_op = post_op

        # Setup feature and ancilla qubits
        self._setup_features()
        self._setup_ancilla()

        super().__init__(qubits + self.feature_qubits, num_layers)

    def circuit(self, *params: Parameters) -> Wires:
        (params,) = params
        n_dim = self.data_qubits.ndim
        data_qubits = self.data_qubits
        data_qubits += [[] for _ in range(len(self.filter_shape))]

        if self.pre_op:  # Pre-op on ancillas
            params = self._filter(self.ancilla_qubits, params)

        # Convolution layers
        for i in range(self.num_layers):  # pylint: disable=unused-variable
            qubits = data_qubits + self.ancilla_qubits

            ### SHIFT
            Convolution.shift(self._filter_shape_qubits, qubits, H=False)

            ### FILTER
            params = self._filter(qubits, params)

            ### PERMUTE
            for j, fsq in enumerate(self._filter_shape_qubits):
                data_qubits[n_dim + j] += data_qubits[j][:fsq]
                data_qubits[j] = data_qubits[j][fsq:]

        if self.post_op:  # Post-op on ancillas
            params = self._filter(self.ancilla_qubits, params)

        # Fully connected layer
        meas = Qubits(
            chain(*zip_longest(self.ancilla_qubits, data_qubits[:n_dim], fillvalue=[]))
        )
        meas += data_qubits[n_dim:] + self.feature_qubits

        meas = meas.flatten()
        self.U_fully_connected(params, meas[::-1])

        return meas[-1]

    @Ansatz.parameter  # pylint: disable=no-member
    def shape(self) -> int:
        n_params = self._n_params * (self.num_layers + self.pre_op + self.post_op)
        n_params += self.U_fully_connected.shape(self.qubits.flatten())

        return n_params

    @property
    def max_layers(self) -> int:
        return min((len(q) for q in self.data_qubits))

    ### PRIVATE

    def _setup_features(self) -> None:
        top = self.data_qubits.total
        self.feature_qubits = [range(top, top + to_qubits(self.num_features))]
        top += to_qubits(self.num_features)

    def _setup_ancilla(self) -> None:
        # TODO: this might be able to be removed
        self.ancilla_qubits = []

        qubits = self.data_qubits
        for i, fsq in enumerate(self._filter_shape_qubits):
            self.ancilla_qubits += [qubits[i][:fsq]]
            qubits[i] = qubits[i][fsq:]
        self.data_qubits = qubits

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

    def _data_wires(self) -> Wires:
        data_qubits = zip_longest(self.ancilla_qubits, self.data_qubits, fillvalue=[])
        return Qubits(chain(*data_qubits)).flatten()

    @property
    def _filter_shape_qubits(self):
        return to_qubits(self.filter_shape)

    @property
    def _n_params(self) -> int:
        return self.num_features * self.U_filter.shape(sum(self._filter_shape_qubits))
