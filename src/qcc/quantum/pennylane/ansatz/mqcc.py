from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from math import sqrt

from qcc.quantum import to_qubits
from qcc.quantum.pennylane import (
    Convolution,
    Multiplex,
    Qubits,
    Unitary,
    QubitsProperty,
)
from qcc.quantum.pennylane.ansatz import Ansatz

from qcc.quantum.pennylane.c2q import ConvolutionAngleFilter
from qcc.quantum.pennylane.fully_connected import FullyConnected
from qcc.quantum.pennylane.convolution import define_filter

if TYPE_CHECKING:
    from typing import Optional
    from pennylane.wires import Wires
    from qcc.quantum.pennylane import Parameters
    from qcc.quantum.pennylane.ansatz.ansatz import Statevector


class MQCC(Ansatz):
    __slots__ = (
        "_data_qubits",
        "_filter_qubits",
        "_feature_qubits",
        "U_filter",
        "U_fully_connected",
        "filter_shape",
        "num_features",
        "pre_op",
        "post_op",
        "pooling",
    )

    data_qubits: Qubits = QubitsProperty(slots=True)
    filter_qubits: Qubits = QubitsProperty(slots=True)
    feature_qubits: Qubits = QubitsProperty(slots=True)

    filter_shape: Iterable[int]
    num_features: int

    U_filter: type[Unitary]
    U_fully_connected: Optional[type[Unitary]]

    pre_op: bool
    post_op: bool

    pooling: bool

    def __init__(
        self,
        qubits: Qubits,
        num_layers: int = None,
        num_features: int = 1,
        filter_shape: Iterable[int] = (2, 2),
        U_filter: type[Unitary] = define_filter(num_layers=4),
        U_fully_connected: Optional[type[Unitary]] = FullyConnected,
        pre_op: bool = False,
        post_op: bool = False,
        pooling: bool = False,
    ):
        self._num_layers = num_layers
        self.num_features = num_features
        self.filter_shape = filter_shape

        self.U_filter = U_filter  # pylint: disable=invalid-name
        self.U_fully_connected = U_fully_connected  # pylint: disable=invalid-name
        self.pre_op = pre_op
        self.post_op = post_op
        self.pooling = pooling

        # Setup feature and ancilla qubits
        qubits = self._setup_qubits(Qubits(qubits))

        super().__init__(qubits, num_layers)

    def circuit(self, *params: Parameters) -> Wires:
        (params,) = params
        fltr_shape_q = to_qubits(self.filter_shape)
        fltr_dim = len(self.filter_shape)
        data_qubits = self.data_qubits
        filter_qubits = [
            self.filter_qubits[fltr_dim * i : fltr_dim * (i + 1)]
            for i in range(self.num_layers)
        ]

        # if self.pre_op:  # Pre-op on ancillas
        #     for ancilla in filter_qubits:
        #         params = self._filter(params, ancilla)

        # Convolution layers
        for i in range(self.num_layers):
            qubits = data_qubits + filter_qubits[i]

            ### SHIFT
            Convolution.shift(fltr_shape_q, qubits)

            ### FILTER
            params = self._filter(params, qubits)

            ### PERMUTE
            Convolution.permute(fltr_shape_q, qubits)

            # TODO: work with assymmetric dims
            if self.pooling and i != self.max_layers - 1:
                for j, fsq in enumerate(fltr_shape_q):
                    data_qubits[j] = data_qubits[j][fsq:]

        # if self.post_op:  # Post-op on ancillas
        #     for ancilla in filter_qubits:
        #         params = self._filter(params, ancilla)

        # Fully connected layer
        meas = data_qubits
        if self.U_fully_connected is not None:
            self.U_fully_connected(params, meas.flatten()[::-1])
            meas = Qubits([meas.flatten()[-1]])

        return (meas + self.filter_qubits).flatten()

    @property
    def shape(self) -> int:
        data_shape_q = self.data_qubits.shape
        fltr_shape_q = to_qubits(self.filter_shape)

        num_params = 0
        for i in range(self.num_layers):
            fsq = (1 for (d, f) in zip(data_shape_q, fltr_shape_q) if d - (i * f))
            num_params += self.U_filter.shape(sum(fsq))

        num_params *= self.num_features

        # TODO: work with assymmetric dims
        if self.U_fully_connected:
            num_qubits = len(self.data_qubits.flatten())
            if self.pooling:
                num_pooling = min(self.max_layers - 1, self.num_layers)
                num_qubits += -sum(fltr_shape_q) * num_pooling

            num_params += self.U_fully_connected.shape(num_qubits)

        return num_params

    def forward(self, psi_in: Optional[Statevector] = None):
        result = super().forward(psi_in)

        # Get subset of output
        norm = sqrt(2**self.filter_qubits.total)
        n = 2 ** (1 + self.feature_qubits.total)
        return norm * result[:, :n]

    @property
    def max_layers(self) -> int:
        dims_qubits = zip(self.data_qubits, to_qubits(self.filter_shape))
        return max((len(q) // max(f, 1) for q, f in dims_qubits))

    ### PRIVATE

    def _setup_qubits(self, qubits: Qubits) -> Qubits:
        # Data qubits
        self.data_qubits = qubits

        # Feature qubits
        top = qubits.total
        self.feature_qubits = [range(top, top + to_qubits(self.num_features))]
        top += self.feature_qubits.total

        # Ancilla qubits
        for i in range(self.num_layers):
            for fsq, dsq in zip(to_qubits(self.filter_shape), self.data_qubits.shape):
                if i * fsq >= dsq:
                    self.filter_qubits += [[]]
                    continue

                self.filter_qubits += [list(range(top, top + fsq))]
                top += fsq

        return self.data_qubits + self.feature_qubits + self.filter_qubits

    @property
    def _data_wires(self) -> Wires:
        return self.data_qubits.flatten()

    def _filter(self, params, qubits: Qubits):
        """Wrapper around self.U_filter that replaces Convolution.filter"""

        # Setup params
        qubits = Qubits(q[:fsq] for q, fsq in zip(qubits, to_qubits(self.filter_shape)))
        num_params = self.num_features * self.U_filter.shape(qubits.total)
        filter_params, params = params[:num_params], params[num_params:]
        shape = (self.num_features, len(filter_params) // self.num_features)
        filter_params = filter_params.reshape(shape)

        # Setup wires
        Multiplex(
            filter_params,
            qubits.flatten(),
            self.feature_qubits.flatten(),
            self.U_filter,
        )

        return params  # Leftover parameters
