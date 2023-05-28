from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires

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
            x = np.reshape(p, (int(len(p) / 2), 2))
            p = np.linalg.norm(x, axis=1)

            with np.errstate(divide="ignore", invalid="ignore"):
                alpha, beta = np.array(
                    [y / m if m > 0 else (1, 0) for y, m in zip(x, p)]
                ).T

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
        op_list = []

        # Keep the type-checker happy
        (params,) = params
        transpose = hyperparameters["transpose"]

        params = list(enumerate(C2Q.get_params(params)))
        if not transpose:
            params = reversed(params)

        for j, (theta, phi, _, t) in params:
            wires_j = wires[j:][::-1]

            if transpose:
                theta = -theta
                phi, t = -t, -phi

            if t.any():
                op_list += [Multiplex(-t, wires_j, qml.RZ)]

            op_list += [Multiplex(theta, wires_j, qml.RY)]

            if phi.any():
                op_list += [Multiplex(phi, wires_j, qml.RZ)]

        return op_list
