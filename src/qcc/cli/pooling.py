from __future__ import annotations
from typing import TYPE_CHECKING

import atexit
import warnings
from enum import StrEnum
from pathlib import Path
from itertools import zip_longest

import numpy as np
from PIL import Image
from polars import DataFrame
from qiskit import QuantumCircuit
from pywt import wavedecn, waverecn

from qcc.file import (
    save,
    filename_labels,
    load_dataframe_from_csv,
    save_dataframe_as_csv,
    save_numpy_as_image,
)
from qcc.quantum import (
    flatten_array,
    normalize,
    to_qubits,
    reconstruct,
    get_fidelity,
    partial_measurement,
    remove_bits,
)
from qcc.quantum.qiskit import QuantumHaarTransform, potato

if TYPE_CHECKING:
    from typing import Iterable


class DimensionReduction(StrEnum):
    NONE = "none"
    AVG = "avg"
    EUCLIDEAN = "euclidean"


def _pooling(
    decomposition_type: DimensionReduction,
    decomposition_levels: Iterable[int],
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
                ("dimension_reduction", str),
                ("noise_free", bool),
                ("fidelity", float),
            ]
        )
    atexit.register(save_dataframe_as_csv, output_dir / "results.csv", df)

    match decomposition_type:
        case DimensionReduction.NONE:
            quantum_dimension_reduction = quantum_no_pooling
            classical_dimension_reduction = classical_no_pooling
        case DimensionReduction.AVG:
            quantum_dimension_reduction = quantum_average_pooling
            classical_dimension_reduction = classical_average_pooling
        case DimensionReduction.EUCLIDEAN:
            quantum_dimension_reduction = quantum_euclidean_pooling
            classical_dimension_reduction = classical_euclidean_pooling
        case _:
            msg = f"Invalid dimension reduction method selected: {decomposition_type}"
            raise AttributeError(msg)

    name = "x".join(str(i) for i in decomposition_levels)
    name = f"{decomposition_type}_{name}"

    for filename in inputs.glob("**/*"):
        try:
            data, mode, suffix, write_fn = _import_file(filename)
        except:
            continue
        if any(2**l > dim for dim, l in zip(data.shape, decomposition_levels)):
            msg = f"Data size {data.shape} not compatible with provided decomposition levels {decomposition_levels}"
            warnings.warn(msg)
            continue
        if (
            any(l != 0 for l in decomposition_levels)
            and decomposition_type == DimensionReduction.NONE
        ):
            msg = f"Without dimension reduction, all decomposition_levels must be 0"
            warnings.warn(msg)
            continue

        dims = "x".join(str(i) for i in data.shape)

        filename = output_dir / "data" / mode / dims / name
        filename = filename_labels(filename, "noise_free" if noise_free else "noisy")
        filename = filename.with_suffix(suffix)

        quantum_data = quantum_dimension_reduction(
            data, decomposition_levels, not noise_free
        )
        if decomposition_type != DimensionReduction.NONE:
            quantum_data = reconstruction(quantum_data, decomposition_levels)

        classical_data = classical_dimension_reduction(data, decomposition_levels)
        if decomposition_type != DimensionReduction.NONE:
            classical_data = reconstruction(classical_data, decomposition_levels)

        # Save results
        save(filename, write_fn(quantum_data))

        classical_filename = filename_labels(filename, "ideal")
        save(classical_filename, write_fn(classical_data))

        quantum_data = np.asarray(quantum_data).flatten(order="F")
        classical_data = np.asarray(data).flatten(order="F")

        # classical_data = classical_data[: np.prod(data.shape[:2])]

        fidelity = get_fidelity(quantum_data, classical_data)
        print(f"{name}, {dims}, {mode}: {fidelity=:.3%}")

        row = DataFrame([[mode, dims, name, noise_free, fidelity]], df.schema)
        df.vstack(row, in_place=True)

        # save_dataframe_as_csv(output_dir / "results.csv", df)


def _import_file(filename: Path):
    data = Image.open(filename, "r")
    mode = data.mode
    if mode == "L":
        mode = "BW"

    data = np.asarray(data, float)

    suffix = ".png"
    write_fn = save_numpy_as_image

    return data, mode, suffix, write_fn


def quantum_no_pooling(
    data: np.ndarray,
    decomposition_levels: Iterable[int],
    noisy_execution: bool = True,
) -> np.ndarray:
    dims = data.shape
    if any(2**l > dim for dim, l in zip(dims, decomposition_levels)):
        raise RuntimeError("uwu")

    psi = flatten_array(data, pad=True)
    psi, mag = normalize(psi, include_magnitude=True)

    dims_q = [to_qubits(x) for x in dims]
    num_qubits = sum(dims_q)

    qc = QuantumCircuit(num_qubits)
    qc.initialize(psi)

    # ==== run ==== #

    psi_out = potato(qc, noisy_execution=noisy_execution)

    # ==== construct image ==== #

    img = reconstruct(mag * psi_out, dims)

    return img


def quantum_average_pooling(
    data: np.ndarray,
    decomposition_levels: Iterable[int],
    noisy_execution: bool = True,
) -> np.ndarray:
    dims = data.shape
    if any(2**l > dim for dim, l in zip(dims, decomposition_levels)):
        raise RuntimeError("uwu")

    psi = flatten_array(data, pad=True)
    psi, mag = normalize(psi, include_magnitude=True)

    dims_q = [to_qubits(x) for x in dims]
    num_qubits = sum(dims_q)

    dims_out = zip_longest(dims, decomposition_levels, fillvalue=0)
    dims_out = [d // (2**l) for d, l in dims_out]

    qc = QuantumCircuit(num_qubits)
    qc.initialize(psi)

    qht = QuantumHaarTransform(dims, decomposition_levels)
    qc.compose(qht, inplace=True)

    # ==== run ==== #

    psi_out = potato(qc, noisy_execution=noisy_execution)

    # ==== construct image ==== #

    img = psi_out.data
    norm = mag / np.sqrt(2 ** sum(decomposition_levels))

    # img_full = reconstruct(norm*img, dims)
    # img_full = Image.fromarray(np.abs(img_full).astype(np.uint8))
    # img_full.save("output_full.png")

    img = reconstruct(norm * img, dims, dims_out)

    return img


def quantum_euclidean_pooling(
    data: np.ndarray,
    decomposition_levels: Iterable[int],
    noisy_execution: bool = True,
):
    dims = data.shape
    if any(2**l > dim for dim, l in zip(dims, decomposition_levels)):
        raise RuntimeError("uwu")

    psi = flatten_array(data, pad=True)
    psi, mag = normalize(psi, include_magnitude=True)

    dims_q = [to_qubits(x) for x in dims]
    num_qubits = sum(dims_q)

    dims_out = zip_longest(dims, decomposition_levels, fillvalue=0)
    dims_out = [d // (2**l) for d, l in dims_out]

    qc = QuantumCircuit(num_qubits)
    qc.initialize(psi)

    base = np.hstack(([0], np.cumsum(dims_q[:-1])))
    top = np.cumsum(dims_q)

    meas = []
    trace = []
    for q, t, l in zip_longest(base, top, decomposition_levels, fillvalue=0):
        trace += list(range(q, q + l))
        meas += qc.qubits[q + l : t]

    # ==== run ==== #

    psi_out = potato(qc, noisy_execution=noisy_execution, meas=meas)

    if not noisy_execution:
        psi_out = partial_measurement(psi_out, trace)
        psi_out = remove_bits(psi_out, trace)
        # psi_out = partial_trace(psi_out, trace)
        # psi_out = np.sqrt(np.diag(psi_out))

    # ==== construct image ==== #

    img = psi_out
    norm = mag / np.sqrt(2 ** sum(decomposition_levels))

    img = reconstruct(norm * img, dims_out)

    return img


def reconstruction(data: np.ndarray, decomposition_levels: Iterable[int]):
    for i, l in enumerate(decomposition_levels):
        for _ in range(l):
            data = [data, dict(d=np.zeros_like(data))]
            data = waverecn(data, "haar", "zero", axes=i) * np.sqrt(2)

    return data


def classical_no_pooling(
    data: np.ndarray,
    decomposition_levels: Iterable[int],
):
    return data


def classical_average_pooling(
    data: np.ndarray,
    decomposition_levels: Iterable[int],
):
    dims = data.shape
    if any(2**l > dim for dim, l in zip(dims, decomposition_levels)):
        raise RuntimeError("uwu")

    for i, l in enumerate(decomposition_levels):
        data = wavedecn(data, "haar", "zero", level=l, axes=i)[0] / np.sqrt(2**l)

    return data


def classical_euclidean_pooling(
    data: np.ndarray,
    decomposition_levels: Iterable[int],
):
    return np.sqrt(classical_average_pooling(data**2, decomposition_levels))
