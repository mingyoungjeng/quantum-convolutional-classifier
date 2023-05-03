"""
ansatz.py: the various ansatz needed for the QCNN
"""

from typing import Any, Sequence, Union
from numbers import Number
from itertools import tee

import numpy as np
import pennylane as qml

# from pennylane import numpy as np


class Ansatz:
    U_params = 3

    def __init__(
        self,
        dims_q: Sequence[int],
        filter_dims: Union[Sequence[int], None] = None,
        stride: int = 1,
        *_,
        **__,
    ) -> None:
        self._dims_q = dims_q
        self.fltr_shape_q = (
            (1, 1)
            if filter_dims is None
            else tuple(int(np.ceil(np.log2(N))) for N in filter_dims)
        )
        self.stride = stride

    @property
    def num_qubits(self):
        return np.sum(self._dims_q)

    @property
    def wires(self):
        return [
            list(range(big - dim, big))
            for big, dim in zip(np.cumsum(self._dims_q), self._dims_q)
        ]

    @staticmethod
    def shift(wires, k: int = 1, control_wires=None):
        if k == 0:
            return

        # Increment / Decrement for k times
        for _ in range(abs(k)):
            for i in range(len(wires))[:: -np.sign(k)]:
                controls = list(wires[:i])

                if control_wires is not None:
                    controls += [control_wires]

                if len(controls) == 0:
                    qml.PauliX(wires[i])
                else:
                    qml.MultiControlledX(wires=controls + [wires[i]])

    @staticmethod
    def U(params: Sequence[Number], wires: Sequence[Number]):
        """
        Applied convolution on high-frequency qubits.

        Args:
            params (Sequence[Number]): rotation gate parameters
            wires (Sequence[Number]): high-frequency qubits of all dimensions
        """

        params = params[: len(wires) * 3]
        params = params.reshape((len(wires), 3))

        # First Rot layer
        for (theta, phi, delta), wire in zip(params, wires):
            qml.Rot(theta, phi, delta, wires=wire)

        # CNOT gates
        control, target = tee(wires)
        first = next(target, None)
        for cnot_wires in zip(control, target):
            qml.CNOT(wires=cnot_wires)

        # Final CNOT gate (last to first)
        if len(wires) > 1:
            qml.CNOT(wires=(wires[-1], first))

    def convolution(self, wires, params):
        ndim = len(self.wires)
        fltr_ndim = len(self.fltr_shape_q)

        # Add extra dimensions if necessary
        for _ in range(fltr_ndim + ndim - len(wires)):
            wires += [[]]

        # Shift operation
        for i, fq in enumerate(self.fltr_shape_q):
            if len(wires[i]) > fq:
                ctrl_qubits, img_qubits = wires[i][:fq], wires[i][fq:]
                wires[i] = ctrl_qubits + img_qubits[fq:]
                wires[ndim + i] += img_qubits[:fq]
            else:
                wires[ndim + i] += wires[i]
                ctrl_qubits = wires[i] = []

            for j, control_qubit in enumerate(ctrl_qubits):
                self.shift(img_qubits[j:], -self.stride, control_wires=control_qubit)

        # Apply convolution filter
        filter_qubits = [
            q for w, fq in zip(wires[ndim:], self.fltr_shape_q) for q in w[-fq:]
        ]
        self.U(params, filter_qubits)

        return wires

    def __call__(self, params, num_layers: int = 1) -> Any:
        wires = self.wires.copy()
        n_params = self.params_per_layer

        # Hybrid convolution/pooling layers
        for _ in range(num_layers):
            conv_params, params = params[:n_params], params[n_params:]
            wires = self.convolution(wires, conv_params)

        # Flatten wires (1D now)
        wires = [q for w in wires for q in w]

        # Fully connected layer
        self.U(params, wires)

        return wires

    @property
    def params_per_layer(self):
        return sum(self.fltr_shape_q) * self.U_params

    def total_params(self, num_layers: int = 1, *_, **__):
        n_params = num_layers * self.params_per_layer
        n_params += self.num_qubits * self.U_params

        return n_params

    @property
    def max_layers(self):
        return np.min(self._dims_q)
