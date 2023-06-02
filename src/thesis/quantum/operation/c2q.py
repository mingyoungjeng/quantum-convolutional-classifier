from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires

from thesis.quantum import flatten_array, normalize
from thesis.quantum.operation import Multiplex

if TYPE_CHECKING:
    from typing import Iterable


class C2Q(Operation):
    num_wires = AnyWires

    def __init__(
        self,
        params,
        wires: Wires,
        transpose: bool = False,
        do_queue=True,
        id=None,
    ):
        self._hyperparameters = {"transpose": transpose}

        super().__init__(params, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def get_params(x_in):
        p = x_in
        while len(p) > 1:
            x = np.reshape(p, (len(p) // 2, 2))
            p = np.linalg.norm(x, axis=1)

            with np.errstate(divide="ignore", invalid="ignore"):
                alpha, beta = zip(*(y / m if m > 0 else (1, 0) for y, m in zip(x, p)))

                alpha_mag, beta_mag = np.abs((alpha, beta))
                alpha_phase, beta_phase = np.angle((alpha, beta))

                theta = 2 * np.arctan(beta_mag / alpha_mag)
                phi = beta_phase - alpha_phase
                r = np.sqrt(alpha_mag**2 + beta_mag**2)
                t = beta_phase + alpha_phase

            yield theta, phi, r, t

    @staticmethod
    def compute_decomposition(
        *params: Iterable,
        wires: Wires,
        **hyperparameters,
    ) -> Iterable[Operation]:
        # Keep the type-checker happy
        (params,) = params
        transpose = hyperparameters["transpose"]

        # Flatten and normalize input state (params)
        # TODO: this has issues with PyTorch
        import torch

        with torch.no_grad():
            params = flatten_array(params, pad=True)
            params, magnitude = normalize(params, include_magnitude=True)
            if magnitude != 1:
                print(f"C2Q parameters were not normalized ({magnitude=}).")

        # Loop setup
        params = list(enumerate(C2Q.get_params(params)))
        if not transpose:
            params = params[::-1]

        # Main C2Q operation
        op_list = []
        for j, (theta, phi, _, t) in params:
            if transpose:
                theta = -theta
                phi, t = -t, -phi

            if t.any():
                op_list += [Multiplex(-t, wires[j], wires[j + 1 :], qml.RZ)]

            op_list += [Multiplex(theta, wires[j], wires[j + 1 :], qml.RY)]

            if phi.any():
                op_list += [Multiplex(phi, wires[j], wires[j + 1 :], qml.RZ)]

        return op_list
