from __future__ import annotations
from typing import TYPE_CHECKING

from numpy import sign
import pennylane as qml
from pennylane.operation import Operation, AnyWires

from qcc.quantum import to_qubits

if TYPE_CHECKING:
    from typing import Iterable
    from pennylane import Wires


class Shift(Operation):
    num_wires = AnyWires

    @property
    def num_params(self) -> int:
        return 1

    @property
    def ndim_params(self) -> tuple[int]:
        return (0,)

    @staticmethod
    def compute_decomposition(*k: int, wires: Wires, **_) -> Iterable[Operation]:
        (k,) = k  # Keep the type-checker happy

        op_list = []
        k, sgn = abs(k), sign(k)

        # ==== calculate the number of bits to use in binary decomposition of shift ==== #
        # k+1 so powers of 2 are represented with the correct number of bits
        #   Ex: 2 = 011, and ceil(log2(3)) = 2
        #       4 = 100, but ceil(log2(4)) = 2
        num_bits = min(to_qubits(k + 1), len(wires))

        # ==== perform shift operation by ±k ==== #
        # Big or little endian depending on which minimizes depth [::sgn]
        for i in range(num_bits)[::sgn]:
            k_i = k // 2**i % 2
            if k_i == 0:
                continue

            # ==== shift by ±1, see [::-sgn] ==== #
            for j, w in tuple(enumerate(wires[i:]))[::-sgn]:
                if j == 0:
                    op_list += [qml.PauliX(w)]
                else:
                    op_list += [qml.MultiControlledX(wires=(*wires[i : i + j], w))]

        return op_list

    def adjoint(self) -> Operation:
        return Shift(-self.parameters, self.wires)
