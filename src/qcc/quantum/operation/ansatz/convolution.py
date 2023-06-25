from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from pennylane.wires import Wires

from qcc.quantum.operation import Convolution, Qubits
from qcc.quantum.operation.ansatz import Ansatz

from qcc.quantum.operation.ansatz.convolution_pooling import ConvolutionPoolingAnsatz

if TYPE_CHECKING:
    from qcc.quantum.operation import Parameters


class ConvolutionAnsatz(ConvolutionPoolingAnsatz):
    def _setup_ancilla(self) -> None:
        self.ancilla_qubits = []
        top = self.qubits.total
        for _ in range(self.num_layers):
            for fsq in self._filter_shape_qubits:
                self.ancilla_qubits += [list(range(top, top + fsq))]
                top += fsq

    @property
    def qubits(self) -> Qubits:
        return self.main_qubits + self.feature_qubits + self.ancilla_qubits

    def circuit(self, *params: Parameters) -> Wires:
        (params,) = params
        fltr_dim = len(self.filter_shape)
        main_qubits = self.main_qubits
        main_qubits += [[] for _ in range(fltr_dim)]
        ancilla_qubits = [
            self.ancilla_qubits[fltr_dim * i : fltr_dim * (i + 1)]
            for i in range(self.num_layers)
        ]

        if self.pre_op:  # Pre-op on ancillas
            for ancilla in ancilla_qubits:
                params = self._filter(ancilla, params)

        # Convolution layers
        for i in range(self.num_layers):
            qubits = main_qubits + ancilla_qubits[i]

            ### SHIFT
            Convolution.shift(self._filter_shape_qubits, qubits)

            ### FILTER
            params = self._filter(qubits, params)

            ### PERMUTE
            Convolution.permute(self._filter_shape_qubits, qubits)

        if self.post_op:  # Post-op on ancillas
            for ancilla in ancilla_qubits:
                params = self._filter(ancilla, params)

        # Fully connected layer
        meas = self.qubits.flatten()
        self.U_fully_connected(params, meas[::-1])

        return meas[-1]

    @Ansatz.parameter  # pylint: disable=no-member
    def shape(self) -> int:
        n_params = self._n_params * (self.num_layers + self.pre_op + self.post_op)
        n_params += self.U_fully_connected.shape(self.qubits.flatten())

        return n_params

    def c2q(self, psi_in, _=None):
        return Ansatz.c2q(self, psi_in=psi_in, wires=self.main_qubits.flatten())

    # def post_processing(self, result):
    #     return super().post_processing(result)[:][
    #         : 2 ** (self.main_qubits.total + self.feature_qubits.total)
    #     ]
