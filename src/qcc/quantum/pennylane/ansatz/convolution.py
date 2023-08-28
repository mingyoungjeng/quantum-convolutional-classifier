from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from qcc.quantum import to_qubits
from qcc.quantum.pennylane import Convolution, Qubits, Unitary
from qcc.quantum.pennylane.ansatz import Ansatz

from qcc.quantum.pennylane.ansatz.convolution_pooling import ConvolutionPoolingAnsatz
from qcc.quantum.pennylane.c2q import ConvolutionAngleFilter
from qcc.quantum.pennylane.fully_connected import FullyConnected

if TYPE_CHECKING:
    from typing import Optional
    from pennylane.wires import Wires
    from qcc.quantum.operation import Parameters
    from qcc.quantum.operation.ansatz.ansatz import Statevector


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

        if self.pre_op:  # Pre-op on ancillas
            for ancilla in ancilla_qubits:
                params = self._filter(params, ancilla)

        # Convolution layers
        for i in range(self.num_layers):
            qubits = data_qubits + ancilla_qubits[i]

            ### SHIFT
            Convolution.shift(fltr_shape_q, qubits)

            ### FILTER
            params = self._filter(params, qubits)

            ### PERMUTE
            Convolution.permute(fltr_shape_q, qubits)

            if self.pooling:
                for i, fsq in enumerate(fltr_shape_q):
                    data_qubits[i] = data_qubits[i][fsq:]

        if self.post_op:  # Post-op on ancillas
            for ancilla in ancilla_qubits:
                params = self._filter(params, ancilla)

        # Fully connected layer
        meas = (data_qubits + ancilla_qubits).flatten()
        if self.U_fully_connected is not None:
            self.U_fully_connected(params, data_qubits.flatten()[::-1])
            return meas[-1]

        return meas

    # @property
    # def shape(self) -> int:
    #     n_params = self._n_params * (self.num_layers + self.pre_op + self.post_op)
    #     n_params += self.U_fully_connected.shape(self.qubits.flatten())

    #     return n_params

    @property
    def _data_wires(self) -> Wires:
        return self.data_qubits.flatten()
    
    def forward(self, psi_in: Optional[Statevector] = None):
        result = super().forward(psi_in)
        return result[:][
            : 2 ** (self.data_qubits.total + self.feature_qubits.total)
        ]
