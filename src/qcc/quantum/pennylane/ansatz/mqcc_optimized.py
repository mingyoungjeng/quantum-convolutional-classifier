from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import chain, zip_longest

from pennylane.ops import Hadamard

from qcc.quantum import to_qubits, parity
from qcc.quantum.pennylane import Convolution, Qubits, QubitsProperty
from qcc.quantum.pennylane.ansatz import Ansatz, MQCC
from qcc.quantum.pennylane.local import define_filter
from qcc.quantum.pennylane.pyramid import Pyramid

if TYPE_CHECKING:
    from typing import Iterable, Optional
    from pennylane.wires import Wires
    from qcc.quantum.pennylane import Parameters, Unitary


class MQCCOptimized(Ansatz):
    __slots__ = (
        "_data_qubits",
        "_filter_qubits",
        "_feature_qubits",
        "U_filter",
        "U_fully_connected",
        "filter_shape",
        "num_features",
        "pooling",
        "pre_op",
        "post_op",
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

    # COMMENT ME
    _filter = MQCC._filter

    def __init__(
        self,
        qubits: Qubits,
        num_layers: int = None,
        num_classes: Optional[int] = 2,
        q2c_method: Ansatz.Q2CMethod | str = Ansatz.Q2CMethod.Probabilities,
        num_features: int = 1,
        filter_shape: Iterable[int] = (2, 2),
        U_filter: type[Unitary] = define_filter(num_layers=4),
        U_fully_connected: Optional[type[Unitary]] = define_filter(num_layers=2),
        pre_op: bool = False,
        post_op: bool = False,
    ):
        self._num_layers = num_layers
        self.num_features = num_features
        self.filter_shape = filter_shape

        self.U_filter = U_filter  # pylint: disable=invalid-name
        self.U_fully_connected = U_fully_connected  # pylint: disable=invalid-name
        self.pre_op = pre_op
        self.post_op = post_op

        if len(filter_shape) > len(qubits):
            msg = f"Filter dimensionality ({len(filter_shape)}) is greater than data dimensionality ({len(qubits)})"
            raise ValueError(msg)

        # Setup feature and ancilla qubits
        qubits = self._setup_qubits(Qubits(qubits))

        super().__init__(qubits, num_layers, num_classes, q2c_method)

    def circuit(self, *params: Parameters) -> Wires:
        (params,) = params
        n_dim = self.data_qubits.ndim
        fltr_shape_q = to_qubits(self.filter_shape)
        data_qubits = self.data_qubits
        data_qubits += [[] for _ in range(len(self.filter_shape))]

        for qubit in self.feature_qubits.flatten():
            Hadamard(wires=qubit)

        if self.pre_op:  # Pre-op on ancillas
            # TODO: fix
            params = self._filter(params, self.filter_qubits)

        # Convolution layers
        for i in range(self.num_layers):  # pylint: disable=unused-variable
            qubits = data_qubits + self.filter_qubits

            ### SHIFT
            Convolution.shift(fltr_shape_q, qubits, H=False)

            ### FILTER
            params = self._filter(params, qubits)

            ### PERMUTE
            for j, fsq in enumerate(fltr_shape_q):
                data_qubits[n_dim + j] += data_qubits[j][:fsq]
                data_qubits[j] = data_qubits[j][fsq:]

        if self.post_op:  # Post-op on ancillas
            params = self._filter(params, self.filter_qubits)

        # Fully connected layer
        controls = zip_longest(data_qubits[n_dim:], self.filter_qubits, fillvalue=[])
        controls = Qubits(sum(*ctrls) for ctrls in controls)
        # print(controls)

        meas = Qubits(data_qubits[:n_dim] + self.feature_qubits)
        # print("meas", meas)

        if self.U_fully_connected is None:
            return meas.flatten()

        meas = Qubits(controls + meas).flatten()
        self.U_fully_connected(params, wires=meas[::-1], num_out=self._num_meas)
        # , num_out=self._num_meas

        match self.q2c_method:
            case self.Q2CMethod.Probabilities:
                return meas[-1]
            case self.Q2CMethod.ExpectationValue:
                return meas[3], meas[-1]
            case self.Q2CMethod.Parity:
                return meas
            case _:
                return

        self.U_fully_connected(params, wires=meas.flatten()[::-1]),
        return meas.flatten()[-1] + controls.flatten()

    @property
    def shape(self) -> int:
        data_shape_q = self.data_qubits.shape
        fltr_shape_q = to_qubits(self.filter_shape)

        num_params = self.pre_op + self.post_op
        num_params *= self.U_filter.shape(sum(fltr_shape_q))

        for i in range(self.num_layers):
            fsq = (f for (d, f) in zip(data_shape_q, fltr_shape_q) if d - (i * f) > 0)
            num_params += self.U_filter.shape(sum(fsq))

        num_params *= self.num_features

        if self.U_fully_connected:
            # num_params += self.U_fully_connected.shape(self._num_meas)
            num_params += self.U_fully_connected.shape(
                self.qubits.flatten(), num_out=self._num_meas
            )
            # , num_out=self._num_meas

        return num_params

    def _forward(self, result):
        return parity(result) if self.U_fully_connected is None else result

    @property
    def max_layers(self) -> int:
        dims_qubits = zip(self.data_qubits, to_qubits(self.filter_shape))
        return max((len(q) // max(f, 1) for q, f in dims_qubits))

    ### PRIVATE

    def _setup_qubits(self, qubits: Qubits) -> Qubits:
        # Feature qubits
        num_feature_qubits = to_qubits(self.num_features)
        self.feature_qubits = [(qubits.total + i for i in range(num_feature_qubits))]

        # Data and ancilla qubits
        fltr_shape_q = to_qubits(self.filter_shape)
        pairs = (([q[fsq:]], [q[:fsq]]) for q, fsq in zip(qubits, fltr_shape_q))
        self.data_qubits, self.filter_qubits = zip(*pairs)

        # Catches scenarios where user defines filter
        # with smaller dimensionality than data
        self.data_qubits += qubits[len(fltr_shape_q) :]

        return qubits + self.feature_qubits

    @property
    def _data_wires(self) -> Wires:
        data_qubits = zip_longest(self.filter_qubits, self.data_qubits, fillvalue=[])
        return Qubits(chain(*data_qubits)).flatten()

    # @property
    # def _num_meas(self) -> int:
    #     data_shape_q = self.data_qubits.shape
    #     fltr_shape_q = to_qubits(self.filter_shape)

    #     fsq = (d - (self.num_layers * f) for d, f in zip(data_shape_q, fltr_shape_q))
    #     fsq = (max(0, q) for q in fsq)

    #     return sum(fsq) + self.feature_qubits.total
