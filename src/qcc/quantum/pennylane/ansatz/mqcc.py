from __future__ import annotations
from typing import TYPE_CHECKING

from torch import sqrt

from pennylane import Hadamard, Select

from qcc.quantum import to_qubits
from qcc.quantum.pennylane import Convolution, Qubits, QubitsProperty
from qcc.quantum.pennylane.ansatz import Ansatz
from qcc.quantum.pennylane.pyramid import Pyramid
from qcc.quantum.pennylane.local import define_filter

if TYPE_CHECKING:
    from typing import Optional, Iterable
    from pennylane.wires import Wires
    from qcc.quantum.pennylane import Parameters, Unitary


class MQCC(Ansatz):
    (
        "_data_qubits",
        "_filter_qubits",
        "_feature_qubits",
        "U_filter",
        "U_fully_connected",
        "filter_shape",
        "num_features",
        "pooling",
    )

    data_qubits: Qubits = QubitsProperty(slots=True)
    filter_qubits: Qubits = QubitsProperty(slots=True)
    feature_qubits: Qubits = QubitsProperty(slots=True)

    filter_shape: Iterable[int]
    num_features: int

    U_filter: type[Unitary]
    U_fully_connected: Optional[type[Unitary]]

    pooling: Iterable[int]

    def __init__(
        self,
        qubits: Qubits,
        num_layers: int = None,
        num_classes: Optional[int] = None,
        q2c_method: Ansatz.Q2CMethod | str = Ansatz.Q2CMethod.Probabilities,
        num_features: int = 1,
        filter_shape: Iterable[int] = (2, 2),
        U_filter: type[Unitary] = define_filter(num_layers=4),
        U_fully_connected: Optional[type[Unitary]] = Pyramid,
        pooling: Iterable[int] | bool = False,
    ):
        self._num_layers = num_layers
        self.num_features = num_features
        self.filter_shape = filter_shape

        self.U_filter = U_filter  # pylint: disable=invalid-name
        self.U_fully_connected = U_fully_connected  # pylint: disable=invalid-name

        if isinstance(pooling, bool):
            pooling = [1 + int(pooling) if f > 1 else 1 for f in filter_shape]
        self.pooling = pooling

        if len(filter_shape) > len(qubits):
            msg = f"Filter dimensionality ({len(filter_shape)}) is greater than data dimensionality ({len(qubits)})"
            raise ValueError(msg)

        # Setup feature and ancilla qubits
        qubits = self._setup_qubits(Qubits(qubits))

        super().__init__(qubits, num_layers, num_classes, q2c_method)

    def circuit(self, *params: Parameters) -> Wires:
        (params,) = params
        fltr_shape_q = to_qubits(self.filter_shape)
        fltr_dim = len(self.filter_shape)
        data_qubits = self.data_qubits
        filter_qubits = [
            self.filter_qubits[fltr_dim * i : fltr_dim * (i + 1)]
            for i in range(self.num_layers)
        ]

        for qubit in self.feature_qubits.flatten():
            Hadamard(wires=qubit)

        # Convolution layers
        for i in range(self.num_layers):
            qubits = data_qubits + filter_qubits[i]

            ### SHIFT
            Convolution.shift(fltr_shape_q, qubits)

            ### FILTER
            params = self._filter(params, qubits)

            ### PERMUTE
            Convolution.permute(fltr_shape_q, qubits)

            ### POOLING
            pooling_q = to_qubits(self.pooling)
            for j, pq in enumerate(pooling_q):
                data_qubits[j] = data_qubits[j][pq:]

        # Fully connected layer
        meas = Qubits(data_qubits + self.feature_qubits)
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
            fsq = (f for (d, f) in zip(data_shape_q, fltr_shape_q) if d - (i * f))
            num_params += self.U_filter.shape(sum(fsq))

        num_params *= self.num_features

        # TODO: work with assymmetric dims
        if self.U_fully_connected:
            num_qubits = len(self.data_qubits.flatten())

            pooling_q = to_qubits(self.pooling)
            num_qubits += -sum(pooling_q) * self.num_layers

            num_params += self.U_fully_connected.shape(num_qubits)

        return num_params

    def _forward(self, result):
        # Get subset of output
        norm = 2**self.filter_qubits.total
        num_states = result.shape[1] // norm
        norm *= 2**self.feature_qubits.total
        # TODO: norm parameter?

        # Adjust norm based on pooling
        pooling_q = to_qubits(self.pooling)
        norm = norm // 2 ** (self.num_layers * sum(pooling_q))

        result = norm * result[:, :num_states]
        result = sqrt(result + 1e-8)

        return result

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

        # Apply filter
        wires = qubits.flatten()
        filters = tuple(self.U_filter(fp, wires=wires) for fp in filter_params)
        Select(filters, self.feature_qubits.flatten())  # TODO: check [::-1]

        return params  # Leftover parameters
