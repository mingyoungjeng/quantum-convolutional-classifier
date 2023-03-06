# from __future__ import annotations
# from typing import TYPE_CHECKING

from typing import Optional, List, Generator, Sequence
import numpy as np
from qiskit_experiments.framework import BaseExperiment, RestlessMixin, BaseAnalysis
from qiskit import QuantumCircuit, Aer, ClassicalRegister
from qiskit.providers import Backend, Options

from numpy.typing import NDArray


# def Ry(theta):
#     return np.matrix(
#         f"{np.cos(theta/2)}, {-np.sin(theta/2)}; {np.sin(theta/2)}, {np.cos(theta/2)}"
#     )


# def Rz(phi):
#     return np.matrix(f"{np.exp(-1j*phi/2)}, {0}; {0}, {np.exp(1j*phi/2)}")


# def Uij(theta=0, phi=0, t=0):
#     return Rz(phi) @ Ry(theta) @ Rz(-t)


class Yeet(BaseExperiment, RestlessMixin):
    """
    _summary_

    Args:
        BaseExperiment (_type_): _description_
        RestlessMixin (_type_): _description_
    """

    def __init__(
        self,
        data: NDArray,
        analysis: Optional[BaseAnalysis] = None,
        backend: Optional[Backend] = None,
    ) -> None:
        """
        Initializes [summary]

        Args:
            data (NDArray): d-dimensional data to be encoded
            D_QHT (NDArray, optional): the number of dimensions to compress. Defaults to the dimensionality of data
            analysis (BaseAnalysis, optional): the analysis to pass to BaseExperiment. Defaults to None.
            backend (Backend, optional): the backend to pass to BaseExperiment. Defaults to None.
        """

        data = np.asarray(data)
        self.dims = data.shape

        # Padding data
        new_dims = [(0, int(2 ** self.requires_qubits(x)) - x) for x in data.shape]
        data = np.pad(data, new_dims, "constant", constant_values=0)

        # Flattening
        psi = data.flatten(order="F")
        num_qubits: int = self.requires_qubits(len(psi))
        self.magnitude = np.linalg.norm(psi)
        self.psi_in = psi / self.magnitude

        if backend is None:
            backend = Aer.get_backend("aer_simulator")

        super().__init__(tuple(range(num_qubits)), analysis, backend)
        # self.set_experiment_options(D_QHT=len(self.dims))

    def circuits(self) -> List[QuantumCircuit]:
        """
        Constructs circuit with C2Q and Q2C steps.

        Returns:
            List[QuantumCircuit]: Single-element list of the QuantumCircuit to be run.
        """
        qc = QuantumCircuit(self.num_qubits)

        # TODO: do this step in get_params
        if self.experiment_options.n_levels > 0:
            self.psi_in = self.partial_measurement()

        self.__c2q__(qc)
        self.__q2c__(qc)
        return [qc]

    def _default_experiment_options(self) -> Options:
        options = super()._default_experiment_options()

        # getattr used to avoid no member error with cls.dims
        options.D_QHT = len(self.dims)
        options.n_levels = 0

        return options

    def high_frequency(self) -> Generator[int, None, None]:
        """
        Returns the high frequency qubits

        Returns:
            _type_: _description_
        """

        options = self.experiment_options

        root = 0
        for i in self.dims[: options.D_QHT]:
            for j in range(root, root + options.n_levels):
                yield j

            root += self.requires_qubits(i)

    def low_frequency(self) -> Generator[int, None, None]:
        """
        Returns the low frequency qubits

        Returns:
            _type_: _description_
        """

        yield from (x for x in range(self.num_qubits) if x not in self.high_frequency())

    def __c2q__(self, qc: QuantumCircuit) -> None:
        theta = []
        phi = []
        t = []

        length = int(2 ** (self.num_qubits - 1))
        for _theta, _phi, _, _t in self.__get_params(self.psi_in):
            _theta, _phi, _t = [
                np.pad(x, (0, length - len(x))) for x in [_theta, _phi, _t]
            ]
            theta.append(_theta)
            phi.append(_phi)
            t.append(_t)
        theta = np.stack(theta, axis=1)
        phi = np.stack(phi, axis=1)
        t = np.stack(t, axis=1)

        low_freq = [*self.low_frequency()]
        low_freq.sort()
        for j in reversed(low_freq):
            control = [qc.qubits[i] for i in filter(lambda x: x > j, low_freq)]
            valid_indices = self.valid_states([c.index - j - 1 for c in control])

            theta_j = np.array([theta[i, j] for i in valid_indices])
            t_j = np.array([t[i, j] for i in valid_indices])
            phi_j = np.array([phi[i, j] for i in valid_indices])

            if t_j.any():
                t_j = -t_j
                qc.ucrz(t_j.tolist(), control, qc.qubits[j])

            qc.ucry(theta_j.tolist(), control, qc.qubits[j])

            if phi_j.any():
                qc.ucrz(phi_j.tolist(), control, qc.qubits[j])

        # U_0 = [
        #     Uij(theta_ij, phi_ij, t_ij)
        #     for (theta_ij, phi_ij, t_ij) in zip(theta_j, phi_j, t_j)
        # ]
        # qc.uc(U_0, qc.qubits[j + 1 :], qc.qubits[j])

    def __q2c__(self, qc: QuantumCircuit) -> None:
        low_freq = [*self.low_frequency()]

        # Consider measure active
        if len(low_freq) == self.num_qubits:
            qc.measure_all()
        else:
            creg = ClassicalRegister(len(low_freq), "meas")
            qc.add_register(creg)
            qc.measure(low_freq, creg)

    # Returns values of theta, phi, r, and t according to Eq. 20
    def __get_params(self, x_in):
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

    @staticmethod
    def requires_qubits(x):
        return int(np.ceil(np.log2(x)))

    @staticmethod
    def valid_states(bits: Sequence[int]) -> Sequence[int]:
        bits.sort()
        values = [0]

        for b in bits:
            values = values + (np.array(values) + (1 << b)).tolist()
        return values

    def partial_measurement(self):
        psi_out = np.zeros_like(self.psi_in)  # Output vector with same length as psi_in

        # Create bitmask from bits_to_remove
        bitmask = sum(2**x for x in self.high_frequency())
        bitmask = ~bitmask  # Negative mask of bitmask

        # Reduce psi_in while preserving indices
        for i, x in enumerate(self.psi_in):
            i_out = i & bitmask  # index in psi_out to insert value
            psi_out[i_out] += (
                np.abs(x) ** 2
            )  # Add the square of the value in psi (probability)
        psi_out = np.sqrt(psi_out)  # Take sqrt to return to a "statevector"

        return psi_out


if __name__ == "__main__":
    pass
