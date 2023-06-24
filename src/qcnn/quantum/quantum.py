from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from astropy.io import fits
import torch
import torch.nn.functional as F
from qcc.ml import create_tensor

if TYPE_CHECKING:
    from typing import Optional, Iterable
    from pathlib import Path


def to_qubits(N: int | Iterable[int]) -> int:
    return np.int_(np.ceil(np.log2(N)))


def normalize(x, include_magnitude=False):
    magnitude = np.linalg.norm(x)
    psi = x / magnitude

    return (psi, magnitude) if include_magnitude else psi


def wires_to_qubits(dims_q, wires=None):
    if wires is None:
        wires = list(range(sum(dims_q)))
    return [wires[x - y : x] for x, y in zip(np.cumsum(dims_q), dims_q)]


def binary(i: int, num_bits: int):
    return tuple(int(b) for b in f"{i:0{num_bits}b}")


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


def pad_array(arr: np.ndarray):
    new_dims = 2 ** to_qubits(arr.shape)
    n_pad = new_dims - arr.shape

    if n_pad.any():  # Don't pad if you don't need to
        n_pad = list(zip([0] * arr.ndim, n_pad))
        arr = np.pad(arr, n_pad, "constant", constant_values=0)

    return arr


def flatten_array(arr: np.ndarray, pad: bool = False):
    if pad:
        arr = pad_array(arr)
    psi = arr.T.ravel()  # == arr.ravel(order="F") # but PyTorch compatible

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
    dp = np.vdot(x_in, x_out)
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


def parity(result, num_classes: int = 2):
    predictions = create_tensor(torch.empty, (len(result), num_classes))

    for i, probs in enumerate(result):
        num_rows = create_tensor(
            torch.tensor, [len(probs) // num_classes] * num_classes
        )
        num_rows[: len(probs) % num_classes] += 1

        pred = F.pad(probs, (0, max(num_rows) * num_classes - len(probs)))
        pred = probs.reshape(max(num_rows), num_classes)
        pred = torch.sum(pred, 0)
        pred /= num_rows
        pred /= sum(pred)

        predictions[i] = pred

    return predictions
