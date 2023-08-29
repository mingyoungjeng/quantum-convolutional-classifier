from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from math import sqrt

from qcc.quantum import to_qubits
from qcc.quantum.pennylane import Convolution, Qubits, Unitary
from qcc.quantum.pennylane.ansatz import Ansatz

from qcc.quantum.pennylane.ansatz.convolution_pooling import ConvolutionPoolingAnsatz
from qcc.quantum.pennylane.c2q import ConvolutionAngleFilter
from qcc.quantum.pennylane.fully_connected import FullyConnected

if TYPE_CHECKING:
    from typing import Optional
    from pennylane.wires import Wires
    from qcc.quantum.pennylane import Parameters
    from qcc.quantum.pennylane.ansatz.ansatz import Statevector


class ConvolutionAnsatz(ConvolutionPoolingAnsatz):
    def __init__(self, *args, pooling: bool = False, **kwargs):
        self.pooling = pooling
        super().__init__(*args, **kwargs)

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
                    self.ancilla_qubits += [[]]
                    continue

                self.ancilla_qubits += [list(range(top, top + fsq))]
                top += fsq

        return self.data_qubits + self.feature_qubits + self.ancilla_qubits

    def circuit(self, *params: Parameters) -> Wires:
        (params,) = params
        fltr_shape_q = to_qubits(self.filter_shape)
        fltr_dim = len(self.filter_shape)
        data_qubits = self.data_qubits
        ancilla_qubits = [
            self.ancilla_qubits[fltr_dim * i : fltr_dim * (i + 1)]
            for i in range(self.num_layers)
        ]

        # if self.pre_op:  # Pre-op on ancillas
        #     for ancilla in ancilla_qubits:
        #         params = self._filter(params, ancilla)

        # Convolution layers
        for i in range(self.num_layers):
            qubits = data_qubits + ancilla_qubits[i]

            ### SHIFT
            Convolution.shift(fltr_shape_q, qubits)

            ### FILTER
            params = self._filter(params, qubits)

            ### PERMUTE
            Convolution.permute(fltr_shape_q, qubits)

            if self.pooling and i != self.max_layers - 1:
                for j, fsq in enumerate(fltr_shape_q):
                    data_qubits[j] = data_qubits[j][fsq:]

        # if self.post_op:  # Post-op on ancillas
        #     for ancilla in ancilla_qubits:
        #         params = self._filter(params, ancilla)

        # Fully connected layer
        meas = data_qubits
        if self.U_fully_connected is not None:
            self.U_fully_connected(params, meas.flatten()[::-1])
            meas = Qubits([meas.flatten()[-1]])

        return (meas + self.ancilla_qubits).flatten()

    @property
    def shape(self) -> int:
        data_shape_q = self.data_qubits.shape
        fltr_shape_q = to_qubits(self.filter_shape)

        num_params = 0
        for i in range(self.num_layers):
            fsq = (1 for (d, f) in zip(data_shape_q, fltr_shape_q) if d - (i * f))
            num_params += self.U_filter.shape(sum(fsq))

        num_params *= self.num_features

        if self.U_fully_connected:  # TODO account for pooling
            num_params += self.U_fully_connected.shape(self.data_qubits.flatten())

        return num_params

    @property
    def _data_wires(self) -> Wires:
        return self.data_qubits.flatten()

    def forward(self, psi_in: Optional[Statevector] = None):
        result = super().forward(psi_in)

        # Get subset of output
        norm = sqrt(2**self.ancilla_qubits.total)
        n = 2 ** (1 + self.feature_qubits.total)
        return norm * result[:, :n]
