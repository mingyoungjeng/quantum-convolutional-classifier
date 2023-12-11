# Depth of quantum convolution

A major advantage of our proposed technique in terms of circuit depth is that the shift and SWAP permutations can be performed for each dimension in parallel.

The primary source of depth is the shift permutations, which for the $i$th dimension of data, requires 1 Hadamard gate to setup and a pyramidal cascade of $n_i$ multi-controlled CNOTs to the shift operator itself.

Multiple-controlled CNOT / Toffoli gate can be implementing with fundamental quantum gates, i.e., single-qubit and CNOT gates, with linear complexity -- $\mathcal{O}(n)$ -- so long as at least 1 ancilla qubit is present [[1]](https://link.springer.com/chapter/10.1007/978-3-031-08760-8_16). Otherwise, the complexity is $\mathcal{O}(2^n)$.

Ancilla qubit(s) are guaranteed with our method of quantum convolution if a) all dimensions of the filter $F$ are greater than 2 pixels, or b) the dimensionality of the data is greater than the dimensionality of the filter, e.g, performing 2-D convolution on 3-D data. If these conditions are not met, the presence of ancilla qubits is dependent on whether the device has ancilla qubits available.

Assuming an ancilla qubit is available, the $n$-qubit shift operation $U_\text{shift}^{-1}$ can be implemented with quadratic complexity:
$$
\sum_{j=1}^{n} \mathcal{O}(j) = \mathcal{O}\left( \frac{n(n + 1)}{2} \right) \approx \mathcal{O}(n^2)
$$

Additionally, for each dimension $i$, a pyramidal cascade of $U_\text{shift}^{-1}$ operations themselves, $n_{f_i}$ in total, are required. However, this does not greatly influence the complexity, since we can assume $n_f \ll n$.
$$
\sum_{j=0}^{n_{f_i} -1} \mathcal{O}((n_i - j)^2) \leq n_{f_i}\mathcal{O}(n_i^2) \approx \mathcal{O}(n_i^2)
$$

The SWAP permutations can be performed across all qubits in parallel, and so thus incurs only a 1 SWAP gate $\Leftrightarrow$ 3 CNOT gate penality to circuit depth.

Since it is based on a method of arbitrary state synthesis, the $U_F$ operation has a depth complexity of $\mathcal{O}(2^{n_f})$ [[2]](https://ieeexplore.ieee.org/abstract/document/10050794). However, since we assume $n_f \ll n$, this typically does not contribute significantly to the overall circuit depth.

Overall, our method can be implemented with complexity $\mathcal{O}(n_{max}^3) + \mathcal{O}(2^{n_f})$, where $n_{max} = \max \{ n_i : 0 \leq i < d \}$ is the number of qubits used to represent the largest dimension of data.
