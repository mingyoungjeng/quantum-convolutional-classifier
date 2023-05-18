import numpy as np
from thesis.quantum import to_qubits
from thesis.quantum.operation.ansatz import SimpleAnsatz


class RapidAnsatz(SimpleAnsatz):
    def circuit(self, params):
        max_wires = np.cumsum(self.num_qubits)
        n_dim = min(self.num_qubits)  # Min number of qubits per dimension

        # Final behavior has different behavior
        for i in reversed(range(1, self.num_layers)):
            # Apply convolution layer:
            idx = np.cumsum([self.convolve.shape()])
            conv_params, params = np.split(params, idx)
            for j in 1 + np.arange(n_dim):
                self.convolve(conv_params, max_wires - j)

            # Apply pooling layers
            half, mod = np.divmod(n_dim, 2)
            for max_wire in max_wires:
                all_wires = range(max_wire - n_dim, max_wire)

                measurements = all_wires[:half][::-1]
                controlled = all_wires[half:][::-1]

                pool_params, params = np.split(params, [self.pool.shape()])
                for m_wire, t_wire in zip(measurements, controlled):
                    self.pool(pool_params, [t_wire, m_wire])
            n_dim = half + mod

        # Qubits to measure
        meas = np.array(
            [np.arange(max_wire - n_dim, max_wire) for max_wire in max_wires]
        ).flatten(order="F")

        # Final layer of convolution
        self.fully_connected(params[: self.fully_connected.shape()], np.sort(meas))

        # Return the minimum required number of qubits to measure in order
        # return np.sort(meas[: int(np.ceil(np.log2(num_classes)))])
        return np.sort(meas)

    @property
    def shape(self):
        n_params = 0
        n_dim = min(self.num_qubits)  # Min number of qubits per dimension

        for i in range(self.num_layers):
            n_params += self.convolve.shape()  # * n_dim  # * len(dims_q)
            half, mod = np.divmod(n_dim, 2)
            n_params += self.pool.shape() * len(self.num_qubits)  # * half
            n_dim = half + mod

        n_params += self.convolve.shape()  # * n_dim * len(dims_q)

        return n_params

    @property
    def max_layers(self) -> int:
        return to_qubits(min(self.num_qubits))
