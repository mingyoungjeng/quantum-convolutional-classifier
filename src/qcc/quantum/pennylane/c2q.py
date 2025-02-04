"""
C2Q Operation in Pennylane

I am overall not happy with this implementation, since there is a lot of breaking gradients.
Rewrite get_params to not break the gradient. Don't use generator, just allocate memory statically.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np
import scipy.linalg as la
import pennylane as qml
from pennylane.operation import Operation, AnyWires

from qcc.ml import USE_CUDA
from qcc.quantum import flatten_array, normalize
from qcc.quantum.pennylane import Unitary

if TYPE_CHECKING:
    from typing import Iterable
    from pennylane.wires import Wires


class C2Q(Operation):
    num_wires = AnyWires

    def __init__(
        self,
        *params,
        wires: Wires,
        angles: bool = False,
        transpose: bool = False,
        id=None,
    ):
        self._hyperparameters = {"transpose": transpose, "angles": angles}

        super().__init__(*params, wires=wires, id=id)

    # TODO: merge with other geT_params in the package
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
                # r = np.sqrt(alpha_mag**2 + beta_mag**2)
                t = beta_phase + alpha_phase

            # yield theta, phi, r, t
            yield theta, phi, t

    @staticmethod
    def permute_angles(params, num_wires):
        idx = lambda x: 2 ** (num_wires - x) * (2**x - 1)
        for j in range(num_wires):
            yield params[idx(j) : idx(j + 1)]

    @staticmethod
    def compute_decomposition(
        *params: Iterable,
        wires: Wires,
        **hyperparameters,
    ) -> Iterable[Operation]:
        # Keep the type-checker happy
        transpose = hyperparameters["transpose"]
        angles = hyperparameters["angles"]

        if angles:  # When passing in angles directly
            if len(params) == 1:
                (params,) = params
                params = params, np.zeros(len(params)), np.zeros(len(params))

            params = zip(*(C2Q.permute_angles(p, len(wires)) for p in params))
        else:  # Flatten and normalize input state
            # TODO: this has issues with PyTorch
            (params,) = params
            params = flatten_array(params, pad=True)
            params, magnitude = normalize(params, include_magnitude=True)
            if magnitude != 1:
                print(f"C2Q parameters were not normalized ({magnitude=}).")
            params = C2Q.get_params(params)

        return C2Q.run(params, wires, transpose)

    @staticmethod
    def run(params, wires, transpose: bool = False):
        # Loop setup
        params = list(enumerate(params))
        if not transpose:
            params = params[::-1]

        # Main C2Q operation
        op_list = []
        for j, (theta, phi, t) in params:
            if transpose:
                theta = -theta
                phi, t = -t, -phi

            if j == 0 and t.any():
                ops = tuple(qml.RZ(-angle, wires=wires[j]) for angle in t)
                op_list += [qml.Select(ops, wires[j + 1 :])]

            ops = tuple(qml.RY(angle, wires=wires[j]) for angle in theta)
            op_list += [qml.Select(ops, wires[j + 1 :])]

            if j == 0 and phi.any():
                ops = tuple(qml.RZ(angle, wires=wires[j]) for angle in phi)
                op_list += [qml.Select(ops, wires[j + 1 :])]

        return op_list


class ConvolutionFilter(Unitary):
    @staticmethod
    def compute_decomposition(params: torch.Tensor, wires, **_):
        # TODO: generalize
        tmp = params.cpu().detach().unsqueeze(0)
        nullspace = torch.tensor(la.null_space(tmp))

        if USE_CUDA:
            nullspace = nullspace.cuda()

        U = torch.hstack((params.unsqueeze(1), nullspace))
        U = U.T

        return [qml.QubitUnitary(U, wires=wires)]

    @staticmethod
    def _shape(num_wires: Wires, **_) -> int:
        return 2**num_wires


class ConvolutionAngleFilter(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        return [C2Q(params, wires=wires, angles=True, transpose=True)]

    @staticmethod
    def _shape(num_wires: Wires, **_) -> int:
        return 2**num_wires - 1


# TODO: format the phi and t terms propertly
class ConvolutionComplexAngleFilter(Unitary):
    @staticmethod
    def compute_decomposition(params, wires, **_):
        num_phi = (len(params) + 1) // 4  # Number of t and phi parameters
        t, theta, phi = params[:num_phi], params[num_phi:-num_phi], params[-num_phi:]

        # janky method of padding
        cls = type(params)
        t, phi = zip(
            *tuple((t[i], phi[i]) if i < num_phi else (0, 0) for i in range(len(theta)))
        )
        return [C2Q(theta, cls(phi), cls(t), wires=wires, angles=True, transpose=True)]

    @staticmethod
    def _shape(num_wires: Wires, **_) -> int:
        return 2 ** (num_wires + 1) - 1
