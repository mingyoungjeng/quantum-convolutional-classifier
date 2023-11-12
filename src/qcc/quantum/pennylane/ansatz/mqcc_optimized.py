from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import chain, zip_longest

import numpy as np
from pennylane import Hadamard, Select
from torch import sqrt, nn, linalg

from qcc.ml import init_params
from qcc.quantum import to_qubits
from qcc.quantum.pennylane import Convolution, Qubits, QubitsProperty
from qcc.quantum.pennylane.ansatz import Ansatz, MQCC
from qcc.quantum.pennylane.local import define_filter

if TYPE_CHECKING:
    from typing import Iterable, Optional
    from pennylane.wires import Wires
    from qcc.quantum.pennylane import Parameters, Unitary
    from qcc.quantum.pennylane.ansatz.ansatz import Statevector
    from torch import Tensor


class MQCCOptimized(Ansatz):
    __slots__ = (
        "_data_qubits",
        "_filter_qubits",
        "_feature_qubits",
        "_class_qubits",
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
    class_qubits: Qubits = QubitsProperty(slots=True)

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
        bias: bool = True,
    ):
        self._num_layers = num_layers
        self.num_features = num_features
        self.num_classes = num_classes
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

        if bias:
            self.register_parameter("bias", init_params(self.num_classes))

        self.register_parameter("norm", init_params(self.num_classes))
        self.reset_parameters()

    def circuit(self, *params: Parameters) -> Wires:
        params, *_ = params
        n_dim = self.data_qubits.ndim
        fltr_shape_q = to_qubits(self.filter_shape)
        data_qubits = self.data_qubits
        data_qubits += [[] for _ in range(len(self.filter_shape))]

        for qubit in self.feature_qubits.flatten():
            Hadamard(wires=qubit)
        for qubit in self.class_qubits.flatten():
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

        in_features = Qubits(data_qubits[:n_dim] + self.feature_qubits)
        # print("in_features", in_features)

        num_params = self.num_classes * self.U_fully_connected.shape(in_features.total)
        fc_params, params = params[:num_params], params[num_params:]
        shape = (self.num_classes, len(fc_params) // self.num_classes)
        fc_params = fc_params.reshape(shape)

        fcs = tuple(
            self.U_fully_connected(p, wires=in_features.flatten()) for p in fc_params
        )
        Select(fcs, self.class_qubits.flatten())

        meas = Qubits(self.class_qubits + in_features + controls).flatten()
        return meas

        # match self.q2c_method:
        #     case self.Q2CMethod.Probabilities:
        #         return meas[-1]
        #     case self.Q2CMethod.ExpectationValue:
        #         return meas[3], meas[-1]
        #     case self.Q2CMethod.Parity:
        #         return meas
        #     case _:
        #         return

        # self.U_fully_connected(params, wires=meas.flatten()[::-1]),
        # return meas.flatten()[-1] + controls.flatten()

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

        # Fully Connected
        num_params += self.num_classes * self.U_fully_connected.shape(self.in_features)

        return num_params

    def forward(self, inputs: Optional[Statevector] = None) -> Tensor:
        if inputs is None:
            return self.qnode(psi_in=inputs)

        # Makes sure batch is 2D array
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        # Normalize inputs
        inputs = inputs.cdouble()  # Fixes issues with Pennylane
        magnitudes = linalg.norm(inputs, dim=1)
        inputs = (inputs.T / magnitudes).T

        result = self.qnode(psi_in=inputs)

        # Unnormalize output
        result = (result.T / magnitudes).T

        # Get subset of output
        norm = 2 ** (self.feature_qubits.total + self.class_qubits.total)
        result = norm * result[:, : self.num_classes]
        result = sqrt(result + 1e-8)

        # Norm and Bias from Fully Connected
        norm = self.get_parameter("norm").unsqueeze(0)
        result = result * norm

        try:  # Apply bias term(s) (if possible)
            bias = self.get_parameter("bias").unsqueeze(0)
            result = result + bias
        except AttributeError:
            pass

        return result.float()

    @property
    def max_layers(self) -> int:
        dims_qubits = zip(self.data_qubits, to_qubits(self.filter_shape))
        return max((len(q) // max(f, 1) for q, f in dims_qubits))

    def reset_parameters(self):
        try:  # Reset bias (if possible)
            k = np.sqrt(1 / self.in_features)
            nn.init.uniform_(self.get_parameter("bias"), -k, k)
        except AttributeError:
            pass

        try:  # Reset magnitude of inverse-c2q parameters (if possible)
            nn.init.uniform_(self.get_parameter("norm"), -1, 1)
        except AttributeError:
            pass

        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    ### PRIVATE

    def _setup_qubits(self, qubits: Qubits) -> Qubits:
        # Feature qubits
        num_feature_qubits = to_qubits(self.num_features)
        self.feature_qubits = [(qubits.total + i for i in range(num_feature_qubits))]

        # Class qubits
        num_class_qubits = to_qubits(self.num_classes)
        self.class_qubits = [
            (qubits.total + num_feature_qubits + i for i in range(num_class_qubits))
        ]

        # Data and ancilla qubits
        fltr_shape_q = to_qubits(self.filter_shape)
        pairs = (([q[fsq:]], [q[:fsq]]) for q, fsq in zip(qubits, fltr_shape_q))
        self.data_qubits, self.filter_qubits = zip(*pairs)

        # Catches scenarios where user defines filter
        # with smaller dimensionality than data
        self.data_qubits += qubits[len(fltr_shape_q) :]

        return qubits + self.feature_qubits + self.class_qubits

    @property
    def _data_wires(self) -> Wires:
        data_qubits = zip_longest(self.filter_qubits, self.data_qubits, fillvalue=[])
        return Qubits(chain(*data_qubits)).flatten()

    @property
    def in_features(self) -> int:
        data_shape_q = self.data_qubits.shape
        fltr_shape_q = to_qubits(self.filter_shape)

        fsq = tuple(
            d - (self.num_layers * f)
            for d, f in zip_longest(data_shape_q, fltr_shape_q, fillvalue=0)
        )
        fsq = tuple(max(0, q) for q in fsq)

        return sum(fsq) + self.feature_qubits.total
