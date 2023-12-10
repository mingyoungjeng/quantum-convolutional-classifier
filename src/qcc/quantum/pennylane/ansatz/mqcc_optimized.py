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


class MQCCOptimized(Ansatz):
    """Quantum-Optimized Multidimensional Quantum Convolutional Classifier"""

    __slots__ = (
        "_data_qubits",
        "_kernel_qubits",
        "_feature_qubits",
        "U_kernel",
        "U_fully_connected",
        "kernel_shape",
        "num_features",
        "pooling",
    )

    data_qubits: Qubits = QubitsProperty(slots=True)
    kernel_qubits: Qubits = QubitsProperty(slots=True)
    feature_qubits: Qubits = QubitsProperty(slots=True)

    kernel_shape: Iterable[int]
    in_channels: int
    out_channels: int

    U_kernel: type[Unitary]
    U_fully_connected: Optional[type[Unitary]]

    pooling: Iterable[int]

    def __init__(
        self,
        qubits: Qubits,
        num_layers: int = None,
        num_classes: Optional[int] = None,
        q2c_method: Ansatz.Q2CMethod | str = Ansatz.Q2CMethod.Probabilities,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_shape: Iterable[int] = (2, 2),
        U_kernel: type[Unitary] = define_filter(num_layers=4),
        U_fully_connected: Optional[type[Unitary]] = Pyramid,
        pooling: Iterable[int] | bool = False,
    ):
        self._num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_shape = kernel_shape

        self.U_kernel = U_kernel  # pylint: disable=invalid-name
        self.U_fully_connected = U_fully_connected  # pylint: disable=invalid-name

        if isinstance(pooling, bool):
            pooling = [1 + int(pooling) if f > 1 else 1 for f in kernel_shape]
        self.pooling = pooling

        if len(kernel_shape) > len(qubits):
            msg = f"Filter dimensionality ({len(kernel_shape)}) is greater than data dimensionality ({len(qubits)})"
            raise ValueError(msg)

        # Setup feature and kernel qubits
        qubits = self._setup_qubits(Qubits(qubits))

        super().__init__(qubits, num_layers, num_classes, q2c_method)

    def circuit(self, *params: Parameters) -> Wires:
        (params,) = params
        kernel_shape_q = to_qubits(self.kernel_shape)
        kernel_dim = len(self.kernel_shape)
        data_qubits = self.data_qubits
        kernel_qubits = [
            self.kernel_qubits[kernel_dim * i : kernel_dim * (i + 1)]
            for i in range(self.num_layers)
        ]

        for qubit in self.feature_qubits.flatten()[to_qubits(self.in_channels) :]:
            Hadamard(wires=qubit)

        # Convolution layers
        for i in range(self.num_layers):
            # ==== pooling ==== #
            pooling_q = to_qubits(self.pooling)
            for j, pq in enumerate(pooling_q):
                data_qubits[j] = data_qubits[j][pq:]

            if data_qubits.total == 0:
                break

            qubits = data_qubits + kernel_qubits[i]

            # ==== shift ==== #
            Convolution.shift(kernel_shape_q, qubits)

            # ==== filter ==== #
            params = self._filter(params, qubits)

            # ==== permute ==== #
            Convolution.permute(kernel_shape_q, qubits)

        # Fully connected layer
        meas = Qubits(data_qubits + self.feature_qubits)
        if self.U_fully_connected is not None:
            self.U_fully_connected(params, meas.flatten()[::-1])
            meas = Qubits([meas.flatten()[-1]])

        return (meas + self.kernel_qubits).flatten()

    @property
    def shape(self) -> int:
        data_shape_q = self.data_qubits.shape
        kernel_shape_q = to_qubits(self.kernel_shape)

        num_params = 0
        for i in range(self.num_layers):
            fsq = (f for (d, f) in zip(data_shape_q, kernel_shape_q) if d - (i + 1 * f))
            num_params += self.U_kernel.shape(sum(fsq))

        num_params *= self.out_channels

        # TODO: work with assymmetric dims
        if self.U_fully_connected:
            num_qubits = len(self.data_qubits.flatten())

            pooling_q = to_qubits(self.pooling)
            num_qubits += -sum(pooling_q) * self.num_layers

            num_params += self.U_fully_connected.shape(num_qubits)

        return num_params

    def _forward(self, result):
        # Get subset of output
        norm = 2**self.kernel_qubits.total
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
        dims_qubits = zip(self.data_qubits, to_qubits(self.kernel_shape))
        return max((len(q) // max(f, 1) for q, f in dims_qubits))

    @property
    def num_features(self) -> int:
        return 2**self.feature_qubits.total

    # ==== private ==== #

    def _setup_qubits(self, qubits: Qubits) -> Qubits:
        # Data qubits
        self.data_qubits = qubits

        # Feature qubits
        top = qubits.total
        self.feature_qubits = [
            range(top, top + to_qubits(max(self.in_channels, self.out_channels)))
        ]
        top += self.feature_qubits.total

        # Kernel qubits
        for i in range(self.num_layers):
            for fsq, dsq in zip(to_qubits(self.kernel_shape), self.data_qubits.shape):
                if i * fsq >= dsq:
                    self.kernel_qubits += [[]]
                    continue

                self.kernel_qubits += [list(range(top, top + fsq))]
                top += fsq

        return self.data_qubits + self.feature_qubits + self.kernel_qubits

    @property
    def _data_wires(self) -> Wires:
        in_channel_qubits = self.feature_qubits.flatten()[: to_qubits(self.in_channels)]
        return self.data_qubits.flatten() + in_channel_qubits

    def _filter(self, params, qubits: Qubits):
        """Wrapper around self.U_kernel that replaces Convolution.filter"""

        # Setup params
        qubits = Qubits(q[:fsq] for q, fsq in zip(qubits, to_qubits(self.kernel_shape)))
        num_params = self.out_channels * self.U_kernel.shape(qubits.total)
        filter_params, params = params[:num_params], params[num_params:]
        shape = (self.out_channels, len(filter_params) // self.out_channels)
        filter_params = filter_params.reshape(shape)

        # Apply filter
        wires = qubits.flatten()
        filters = tuple(self.U_kernel(fp, wires=wires) for fp in filter_params)
        Select(filters, self.feature_qubits.flatten()[: to_qubits(self.out_channels)])

        return params  # Leftover parameters
