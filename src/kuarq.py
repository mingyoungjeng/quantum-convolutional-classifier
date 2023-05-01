from typing import Optional, Union
from pathlib import Path
import numpy as np
from PIL import Image
from astropy.io import fits


def to_qubits(length: int) -> int:
    return int(np.ceil(np.log2(length)))


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


def flatten_image(
    filename: Path,
    include_mode: Optional[bool] = False,
    multispectral: Optional[bool] = False,
    pad=False,
):  # -> tuple[npt.NDArray[np.complex64], *[tuple[int, ...]]]:
    with fits.open(filename) if multispectral else Image.open(filename, "r") as im:
        matrix = im[0].data if multispectral else np.asarray(im, dtype=float)
        dims = matrix.shape
        if pad:
            new_dims = [(0, int(2 ** np.ceil(np.log2(x))) - x) for x in matrix.shape]
            matrix = np.pad(matrix, new_dims, "constant", constant_values=0)
        psi = matrix.flatten(order="F")
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


# TODO: make sure this doesn't throw error when parent doesn't exist
def create_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def export_img(img: Union[Image.Image, fits.HDUList], path: Path):
    if isinstance(img, fits.HDUList):
        path = path.with_suffix(".fits")

    create_parent(path)

    filename = path.stem

    i = 1
    while path.is_file():
        # path = path.with_stem(f"{filename}_{i}")
        path = path.with_name(f"{filename}_{i}{path.suffix}")
        i += 1

    if isinstance(img, fits.HDUList):
        img.writeto(path)
        return

    img.save(path)
