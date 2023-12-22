"""
Convolution CLI

Taken from entropy2023. Might not work.

TODO: implement
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import atexit
from enum import StrEnum
from pathlib import Path
from functools import partial

import numpy as np
from PIL import Image
import soundfile as sf
from librosa.display import waveshow
from spectral import save_rgb
import matplotlib.pyplot as plt
from polars import DataFrame
from qiskit import QuantumCircuit

from qcc.file import (
    save,
    filename_labels,
    load_dataframe_from_csv,
    save_dataframe_as_csv,
)
from qcc.filters import (
    convolution,
    avg_filter,
    sobel_filter,
    gaussian_blur,
    laplacian_approx,
    update_dims,
)
from qcc.quantum import (
    flatten_array,
    normalize,
    to_qubits,
    reconstruct,
    get_fidelity,
)
from qcc.quantum.qiskit import Convolution, potato

if TYPE_CHECKING:
    from typing import Sequence


class Filters(StrEnum):
    AVG = "avg"
    SOBEL_X = "sobel-x"
    SOBEL_Y = "sobel-y"
    GAUSSIAN = "gaussian"
    LAPLACIAN = "laplacian"


def _convolution(
    filter_name,
    filter_dims: Sequence[int],
    inputs: Path,
    output_dir: Path,
    noise_free: bool,
):
    df = load_dataframe_from_csv(output_dir / "results.csv")
    if df is None:
        df = DataFrame(
            schema=[
                ("mode", str),
                ("data_size", str),
                ("filter", str),
                ("noise_free", bool),
                ("fidelity", float),
            ]
        )
    atexit.register(save_dataframe_as_csv, output_dir / "results.csv", df)

    # TODO: this is temp
    filter_size = filter_dims[0]
    dim = len(filter_dims)

    match filter_name:
        case Filters.AVG:
            kernel = avg_filter(filter_size, dim=dim)
        case Filters.SOBEL_X:
            kernel = sobel_filter(filter_size, axis=1, dim=dim)
        case Filters.SOBEL_Y:
            kernel = sobel_filter(filter_size, axis=0, dim=dim)
        case Filters.LAPLACIAN:
            kernel = laplacian_approx(filter_size, dim=dim)
        case Filters.GAUSSIAN:
            kernel = gaussian_blur(filter_size, dim=dim)
        case _:
            raise AttributeError(f"Invalid filter selected: {filter_name}")

    filter_dims = "x".join(str(i) for i in filter_dims)
    name = f"{filter_name}_{filter_dims}"

    for filename in inputs.glob("**/*"):
        try:
            data, mode, suffix, write_fn = _import_file(filename)
        except:
            continue

        dims = "x".join(str(i) for i in data.shape)

        filename = output_dir / "data" / mode / dims / f"{filename.stem}_{name}"
        filename = filename_labels(filename, "noise_free" if noise_free else "noisy")
        filename = filename.with_suffix(suffix)

        quantum_data = quantum_convolution(data, kernel, not noise_free)
        classical_data = convolution(data, kernel)
        # print(quantum_data.shape, classical_data.shape)

        # Save results
        save(filename, write_fn(quantum_data))

        classical_filename = filename_labels(filename, "ideal")
        save(classical_filename, write_fn(classical_data))

        quantum_data = np.asarray(quantum_data).flatten(order="F")
        classical_data = np.asarray(classical_data).flatten(order="F")

        # classical_data = classical_data[: np.prod(data.shape[:2])]

        fidelity = get_fidelity(quantum_data, classical_data)
        print(f"{name}, {dims}, {mode}: {fidelity=:.3%}")

        row = DataFrame([[mode, dims, name, noise_free, fidelity]], df.schema)
        df.vstack(row, in_place=True)


def _import_file(filename: Path):
    # ==== rgb images ==== #
    try:
        data = Image.open(filename, "r")
        mode = data.mode
        if mode == "L":
            mode = "BW"

        data = np.asarray(data, float)

        suffix = ".png"
        write_fn = _write_image

        return data, mode, suffix, write_fn
    except IOError:
        pass

    # ==== audio ==== #
    try:
        with open(filename, "rb") as f:
            data, samplerate = sf.read(f)

        mode = "audio"
        suffix = ".png"
        write_fn = partial(_write_audio_as_img, samplerate=samplerate)

        # suffix = ".flac"
        # write_fn = partial(_write_audio, samplerate=samplerate)

        return data, mode, suffix, write_fn
    except (TypeError, RuntimeError):
        pass

    # ==== hyperspectral images ==== #
    try:
        data = np.load(filename).astype(float)
        mode = "hyperspectral"
        suffix = ".png"
        write_fn = _write_hyperspectral_as_img

        # suffix = ".npy"
        # write_fn = _write_hyperspectral

        return data, mode, suffix, write_fn
    except Exception as e:
        print(e)

    raise RuntimeError(f"No valid file input found for {filename}")


def quantum_convolution(
    data: np.ndarray,
    kernel: np.ndarray,
    noisy_execution: bool = True,
):
    dims = data.shape
    dims_out = update_dims(dims, kernel_size=kernel.shape, stride=1)

    npad = tuple((0, 2 ** int(np.ceil(np.log2(N))) - N) for N in kernel.shape)
    kernel = np.pad(kernel, pad_width=npad, mode="constant", constant_values=0)

    psi = flatten_array(data, pad=True)
    psi, mag = normalize(psi, include_magnitude=True)

    dims_q = [to_qubits(x) for x in dims]

    num_qubits = sum(dims_q)
    num_states = 2**num_qubits

    kernel_shape_q = [to_qubits(filter_size) for filter_size in kernel.shape]
    num_kernel_qubits = sum(kernel_shape_q)
    total_qubits = num_qubits + num_kernel_qubits

    qc = QuantumCircuit(total_qubits)
    qc.initialize(psi, qc.qubits[:-num_kernel_qubits])

    convolution_gate = Convolution(dims, kernel)
    qc.compose(convolution_gate, inplace=True)

    # ==== run ==== #

    psi_out = potato(qc, noisy_execution=noisy_execution, shots=32000)

    # dims = dims[:2]
    # num_states = np.prod(dims)

    # ==== Construct image ==== #
    i = 0
    data = psi_out.data[i * num_states : (i + 1) * num_states]
    norm = mag * convolution_gate.kernel_norm * np.sqrt(2**num_kernel_qubits)

    data = reconstruct(norm * data, dims, dims_out)

    return data


def _write_image(data: np.ndarray):
    # Convert data type
    data = np.abs(data).astype(np.uint8)

    # Create Image
    img = Image.fromarray(data)

    return img.save


def _write_audio(data: np.ndarray, samplerate: int = 44100):
    # Make sure data is 2D
    if len(data.shape) == 1:
        data = np.expand_dims(data, 1)

    fn = lambda data: partial(
        sf.write,
        data=data,
        samplerate=samplerate,
    )

    # WHY CAN YOU NOT ACCEPT PATHLIB
    return lambda filename: fn(str(filename))


def _write_audio_as_img(data: np.ndarray, samplerate: int = 44100):
    plt.cla()
    plt.figure(figsize=(4, 4))
    waveshow(data.astype(float), sr=samplerate, color="#37b6f0")
    plt.axis("off")

    return partial(plt.savefig, dpi=300, bbox_inches="tight", pad_inches=0)


def _write_hyperspectral(data: np.ndarray):
    return partial(np.save, arr=data)


def _write_hyperspectral_as_img(data: np.ndarray):
    return partial(save_rgb, data=data)
