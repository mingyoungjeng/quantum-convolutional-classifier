import numpy as np
from qiskit import QuantumCircuit
from kuarq import to_qubits, normalize


def get_params(x_in):
    p = x_in
    while len(p) > 1:
        x = np.reshape(p, (int(len(p) / 2), 2))
        p = np.linalg.norm(x, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            alpha, beta = np.array([y / m if m > 0 else (1, 0) for y, m in zip(x, p)]).T

            alpha_mag, beta_mag = np.abs((alpha, beta))
            alpha_phase, beta_phase = np.angle((alpha, beta))

            with np.errstate(divide="ignore"):
                theta = 2 * np.arctan(beta_mag / alpha_mag)
            phi = beta_phase - alpha_phase
            r = np.sqrt(alpha_mag**2 + beta_mag**2)
            t = beta_phase + alpha_phase

        yield theta, phi, r, t


def c2q(qc: QuantumCircuit, psi_in, targets=None, transpose=False) -> None:
    if targets is None:
        targets = list(range(qc.num_qubits))

    theta = []
    phi = []
    t = []

    length = int(2 ** (len(targets) - 1))
    for _theta, _phi, _, _t in get_params(psi_in):
        _theta, _phi, _t = [np.pad(x, (0, length - len(x))) for x in [_theta, _phi, _t]]
        theta.append(_theta)
        phi.append(_phi)
        t.append(_t)
    theta = np.stack(theta, axis=1)
    phi = np.stack(phi, axis=1)
    t = np.stack(t, axis=1)

    if transpose:
        theta = -theta
        phi, t = -t, -phi

    targets = targets[:: 1 if transpose else -1]
    for j, target in enumerate(targets):
        control = [qc.qubits[t] for t in targets[j + 1 :]]

        n_j = len(targets) - 1 - j
        i_max = 2 ** (n_j)

        theta_j = theta[:i_max, j]
        t_j = t[:i_max, j]
        phi_j = phi[:i_max, j]

        if t_j.any():
            t_j = -t_j
            qc.ucrz(t_j.tolist(), control, qc.qubits[target])

        qc.ucry(theta_j.tolist(), control, qc.qubits[target])

        if phi_j.any():
            qc.ucrz(phi_j.tolist(), control, qc.qubits[target])


def shift(qc: QuantumCircuit, k: int = 1, targets=None, control=None) -> None:
    if k == 0:
        return
    if targets is None:
        targets = qc.qubits

    # Increment / Decrement for
    for _ in range(abs(k)):
        for i in range(len(targets))[:: -np.sign(k)]:
            controls = list(targets[:i])

            if control is not None:
                controls += [control]

            if len(controls) == 0:
                qc.x(targets[i])
            else:
                qc.mcx(controls, targets[i])


def rotate(qc: QuantumCircuit, start: int, end: int, direction: str = "right") -> None:
    rng = range(start, end)

    if direction == "left":
        rng = reversed(rng)

    for y in rng:
        qc.swap(y, y + 1)


def conv_pool_hybrid(
    qc: QuantumCircuit, wires, fltr, stride: int = 1, n_dim: int = None
):
    if n_dim is None:
        n_dim = len(wires)

    if len(wires) < n_dim + fltr.ndim:
        wires.extend([[] for _ in range(fltr.ndim + n_dim - len(wires))])
    fltr_shape_q = [to_qubits(filter_size) for filter_size in fltr.shape]

    params, fltr_mag = normalize(fltr.flatten(order="F"), include_magnitude=True)

    ### Shift operation
    controls = [[] for _ in range(fltr.ndim)]
    for i, fq in enumerate(fltr_shape_q):
        ctrl_qubits, img_qubits = wires[i][:fq], wires[i][fq:]
        wires[i] = img_qubits
        controls[i] = ctrl_qubits

        for j, control_qubit in enumerate(ctrl_qubits):
            shift(qc, -stride, targets=img_qubits[j:], control=control_qubit)

    ### Filter using C2Q
    filter_qubits = np.array([wires_d[:fq] for wires_d, fq in zip(wires, fltr_shape_q)])
    c2q(qc, params, targets=filter_qubits.flatten(), transpose=True)

    for j, (ctrls, trgts) in enumerate(zip(controls, wires)):
        wires[n_dim + j].extend(ctrls)
        for c, t in zip(ctrls, trgts):
            qc.swap(c, t)

    return wires, fltr_mag
