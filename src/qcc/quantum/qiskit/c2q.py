from __future__ import annotations
from typing import TYPE_CHECKING

from itertools import count

import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.parameterexpression import ParameterValueType

from qiskit.exceptions import QiskitError

if TYPE_CHECKING:
    from typing import Optional


class C2Q(Gate):
    instances_counter = count()

    @property
    def _id(self):
        i = next(self.instances_counter)
        return str() if i == 0 else i

    def __init__(self, num_qubits: int, label: Optional[str] = None) -> None:
        length = 2**num_qubits
        params_name = f"c2q{self._id}"
        params = ParameterVector(params_name, length=length)

        super().__init__("C2Q", num_qubits, params, label)

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

                with np.errstate(divide="ignore"):
                    theta = 2 * np.arctan(beta_mag / alpha_mag)
                phi = beta_phase - alpha_phase
                r = np.sqrt(alpha_mag**2 + beta_mag**2)
                t = beta_phase + alpha_phase

            yield theta, phi, r, t

    def _define(self):
        gate = C2QAnsatz(self.num_qubits)
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        qc.compose(gate, inplace=True)

        # Makes sures parameters are assigned
        params = np.array(self.params)
        if any(isinstance(p, ParameterValueType) for p in params):
            self.definition = qc
            return

        angle_params = np.empty(qc.num_parameters)
        for i, (theta, phi, _, t) in enumerate(self.get_params(params)):
            if i == 0:
                angle_params[: gate.index_map(1)] = np.concatenate((-t, phi))

            angle_params[gate.index_map(i + 1) : gate.index_map(i + 2)] = theta

        self.definition = qc.assign_parameters(angle_params)


class C2QAnsatz(Gate):
    instances_counter = count()

    @property
    def _id(self):
        i = next(self.instances_counter)
        return str() if i == 0 else i

    # TODO: positive, real vs general
    def __init__(self, num_qubits: int, label: Optional[str] = None) -> None:
        length = 2 ** (num_qubits + 1) - 1
        params_name = f"c2q_angle{self._id}"
        params = ParameterVector(params_name, length=length)

        super().__init__("C2Q Angles", num_qubits, params, label)

    def _define(self):
        qc = QuantumCircuit(self.num_qubits, name=self.name)

        for i, q in enumerate(qc.qubits[::-1]):
            j = self.num_qubits - 1 - i

            if j == 0:  # Last iteration
                phases = self.params[: self.index_map(j + 1)]
                t = phases[: len(phases) // 2]
                phi = phases[len(phases) // 2 :]

                self.multiplex(qc, t, qc.qubits[j:], rot_axis="T")

            theta = self.params[self.index_map(j + 1) : self.index_map(j + 2)]
            self.multiplex(qc, theta, qc.qubits[j:], rot_axis="Y")

            if j == 0:
                self.multiplex(qc, phi, qc.qubits[j:], rot_axis="Z")

        self.definition = qc

    def index_map(self, x):
        return 2 ** (self.num_qubits + 1 - x) * (2**x - 1)

    @staticmethod
    def multiplex(qc: QuantumCircuit, params, qubits, rot_axis="Y"):
        if all(p == 0 for p in params):
            return

        gate = QuantumCircuit(len(qubits))
        ucrot = MultiplexAnsatz(len(qubits), rot_axis=rot_axis)
        gate.compose(ucrot, inplace=True)
        gate.assign_parameters(params, inplace=True)

        qc.compose(gate, qubits, inplace=True)


# Modified from qiskit.extensions.quantum_initializer.UCPauliRotGate
class MultiplexAnsatz(Gate):
    instances_counter = count()

    @property
    def _id(self):
        i = next(self.instances_counter)
        return str() if i == 0 else i

    def __init__(self, num_qubits: int, rot_axis):
        self.rot_axes = rot_axis
        if rot_axis not in ("X", "Y", "Z", "T"):
            raise QiskitError("Rotation axis is not supported.")

        name = f"ucr{rot_axis.lower()}"
        params_name = f"{name}{self._id}"
        params = ParameterVector(params_name, length=2 ** (num_qubits - 1))
        super().__init__(name, num_qubits, params)

    def _define(self):
        """
        Finds a decomposition of a UC rotation gate into elementary gates
        (C-NOTs and single-qubit rotations).
        """
        qc = QuantumCircuit(self.num_qubits)
        q_target = qc.qubits[0]
        q_controls = qc.qubits[1:]
        if not q_controls:  # equivalent to: if len(q_controls) == 0
            if self.rot_axes == "X":
                qc.rx(self.params[0], q_target)
            if self.rot_axes == "Y":
                qc.ry(self.params[0], q_target)
            if self.rot_axes == "Z":
                qc.rz(self.params[0], q_target)
            if self.rot_axes == "T":
                qc.rz(-self.params[0], q_target)
        else:
            # Now, it is easy to place the C-NOT gates to get back the full decomposition.
            for i, angle in enumerate(self.params):
                if self.rot_axes == "X":
                    qc.rx(angle, q_target)
                if self.rot_axes == "Y":
                    qc.ry(angle, q_target)
                if self.rot_axes == "Z":
                    qc.rz(angle, q_target)
                if self.rot_axes == "T":
                    qc.rz(-angle, q_target)
                # Determine the index of the qubit we want to control the C-NOT gate.
                # Note that it corresponds
                # to the number of trailing zeros in the binary representation of i+1
                if not i == len(self.params) - 1:
                    binary_rep = np.binary_repr(i + 1)
                    q_contr_index = len(binary_rep) - len(binary_rep.rstrip("0"))
                else:
                    # Handle special case:
                    q_contr_index = len(q_controls) - 1
                # For X rotations, we have to additionally place some Ry gates around the
                # C-NOT gates. They change the basis of the NOT operation, such that the
                # decomposition of for uniformly controlled X rotations works correctly by symmetry
                # with the decomposition of uniformly controlled Z or Y rotations
                if self.rot_axes == "X":
                    qc.ry(np.pi / 2, q_target)
                qc.cx(q_controls[q_contr_index], q_target)
                if self.rot_axes == "X":
                    qc.ry(-np.pi / 2, q_target)

        self.definition = qc
