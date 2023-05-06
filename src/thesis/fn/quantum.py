from __future__ import annotations
from typing import TYPE_CHECKING

from functools import cache
import numpy as np
from PIL import Image
from astropy.io import fits

if TYPE_CHECKING:
    from typing import Optional, Sequence
    from pathlib import Path
    from qiskit import QuantumCircuit


@cache
def to_qubits(N: int | Sequence[int]) -> int:
    return np.int_(np.ceil(np.log2(N)))


def normalize(x, include_magnitude=False):
    magnitude = np.linalg.norm(x)
    psi = x / magnitude

    return (psi, magnitude) if include_magnitude else psi


def random_data(
    qubits: int, is_complex: Optional[bool] = False, seed: Optional[float] = None
):
    if seed is not None:
        np.random.seed(seed)

    n_states: int = 2**qubits
    psi = np.random.rand(n_states)

    if is_complex:
        psi = psi + (np.random.rand(n_states) * 2j) - 1j
    else:
        psi = psi * 255

    return psi


def flatten_array(arr: np.ndarray, pad: bool = False):
    if pad:
        new_dims = 2 ** to_qubits(arr.shape)
        n_pad = list(zip([0] * arr.ndim, new_dims - arr.shape))
        arr = np.pad(arr, n_pad, "constant", constant_values=0)
    psi = np.ravel(arr, order="F")

    return psi


def flatten_image(
    filename: Path,
    include_mode: Optional[bool] = False,
    multispectral: Optional[bool] = False,
    pad=False,
):  # -> tuple[npt.NDArray[np.complex64], *[tuple[int, ...]]]:
    with fits.open(filename) if multispectral else Image.open(filename, "r") as im:
        matrix = im[0].data if multispectral else np.asarray(im, dtype=float)
        dims = matrix.shape
        psi = flatten_array(matrix, pad)
        mode = "fits" if multispectral else im.mode

    if include_mode:
        return psi, mode, *dims

    return psi, *dims


def from_counts(
    counts: dict,
    shots: int,
    num_qubits: Optional[int] = None,
    reverse_bits: bool = False,
):
    num_states: int = 2 ** (
        to_qubits(len(counts)) if num_qubits is None else num_qubits
    )
    psi_out = np.zeros(num_states)

    for bit, count in counts.items():
        if reverse_bits:
            bit = bit[::-1]
        psi_out[int(bit, 2)] = int(count) / shots

    return np.sqrt(psi_out)


def get_fidelity(x_in, x_out) -> float:
    # for x, y in zip(x_in, x_out):
    #     print(x, y)
    x_in = normalize(x_in)
    x_out = normalize(x_out)
    dp = np.dot(x_in, x_out)
    fidelity = np.abs(dp) ** 2
    return fidelity


def construct_img(img_data, size, mode: str = None) -> Image.Image:
    new_dims = [2 ** to_qubits(x) for x in size]
    image = np.abs(img_data).astype(np.uint8)  # Image package needs uint8
    image = image.reshape(new_dims, order="F")
    image = image[tuple([slice(s) for s in size])]  # Remove padded zeroes

    if mode == "fits":
        hdu = fits.PrimaryHDU(image)
        return fits.HDUList([hdu])

    return Image.fromarray(image, mode)


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
