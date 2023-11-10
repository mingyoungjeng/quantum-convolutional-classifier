from __future__ import annotations
from typing import TYPE_CHECKING

import atexit
from enum import StrEnum
from pathlib import Path
from functools import partial

import click

import numpy as np
from PIL import Image
import soundfile as sf
from librosa.display import waveshow
from spectral import save_rgb
import matplotlib.pyplot as plt
from polars import DataFrame
from qiskit import QuantumCircuit, Aer, execute

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
    laplacian,
)
from qcc.quantum import (
    flatten_array,
    normalize,
    to_qubits,
    from_counts,
    reconstruct,
    get_fidelity,
)

if TYPE_CHECKING:
    pass


@click.group()
@click.pass_context
def cli(ctx: click.Context):
    # Set context settings
    ctx.show_default = True


def create_results(ctx, param, value):
    results_dir = value / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


class Filters(StrEnum):
    AVG = "avg"
    SOBEL_X = "sobel-x"
    SOBEL_Y = "sobel-y"
    GAUSSIAN = "gaussian"
    LAPLACIAN = "laplacian"


@cli.command()
@click.pass_context
@click.option(
    "--filter",
    "filter_name",
    type=click.Choice(
        [e.value for e in Filters],
        case_sensitive=False,
    ),
)
@click.option(
    "--filter_size",
    type=int,
    default=3,
)
@click.option(
    "--dim",
    type=int,
    default=2,
)
@click.option(
    "-i",
    "--inputs",
    required=True,
    type=click.Path(
        exists=True,
        path_type=Path,
        resolve_path=True,
        file_okay=False,
        writable=True,
    ),
    default=Path.cwd() / "data",
    show_default=False,
    help="Input data",
)
@click.option(
    "-o",
    "--output_dir",
    required=True,
    type=click.Path(
        exists=True,
        path_type=Path,
        resolve_path=True,
        file_okay=False,
        writable=True,
    ),
    default=Path.cwd(),
    show_default=False,
    callback=create_results,
    help="Output directory",
)
@click.option(
    "--noiseless/--noisy",
    default=True,
    required=True,
)
def run(
    ctx,
    filter_name,
    filter_size: int,
    dim: int,
    inputs: Path,
    output_dir: Path,
    noiseless: bool,
):
    df = load_dataframe_from_csv(output_dir / "results.csv")
    if df is None:
        df = DataFrame(
            schema=[
                ("mode", str),
                ("data_size", str),
                ("filter", str),
                ("noiseless", bool),
                ("fidelity", float),
            ]
        )
    atexit.register(save_dataframe_as_csv, output_dir / "results.csv", df)

    match filter_name:
        case Filters.AVG:
            fltr = avg_filter(filter_size, dim=dim)
        case Filters.SOBEL_X:
            fltr = sobel_filter(filter_size, axis=1, dim=dim)
        case Filters.SOBEL_Y:
            fltr = sobel_filter(filter_size, axis=0, dim=dim)
        case Filters.LAPLACIAN:
            fltr = laplacian(filter_size, dim=dim)
        case Filters.GAUSSIAN:
            fltr = gaussian_blur(filter_size, dim=dim)
        case _:
            raise AttributeError(f"Invalid filter selected: {filter_name}")

    filter_dims = "x".join(str(i) for i in fltr.shape)
    name = f"{filter_name}_{filter_dims}"

    for filename in inputs.glob("**/*"):
        try:
            data, mode, suffix, write_fn = _import_file(filename)
        except:
            continue

        dims = "x".join(str(i) for i in data.shape)

        filename = output_dir / "data" / mode / dims / name
        filename = filename_labels(filename, "noiseless" if noiseless else "noisy")
        filename = filename.with_suffix(suffix)

        quantum_data = quantum_convolution(data, fltr, not noiseless)
        classical_data = convolution(data, fltr)

        # Save results
        save(filename, write_fn(quantum_data))

        classical_filename = filename_labels(filename, "ideal")
        save(classical_filename, write_fn(classical_data))

        quantum_data = np.asarray(quantum_data).flatten(order="F")
        classical_data = np.asarray(classical_data).flatten(order="F")

        # classical_data = classical_data[: np.prod(data.shape[:2])]

        fidelity = get_fidelity(quantum_data, classical_data)
        print(f"{name}, {dims}, {mode}: {fidelity=:.3%}")

        row = DataFrame([[mode, dims, name, noiseless, fidelity]], df.schema)
        df.vstack(row, in_place=True)


def _import_file(filename: Path):
    try:  # Images
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

    try:  # Audio
        with open(filename, "rb") as f:
            data, samplerate = sf.read(f)

        mode = "audio"
        # suffix = ".png"
        # write_fn = partial(_write_audio_as_img, samplerate=samplerate)
        
        suffix = ".flac"
        write_fn = partial(_write_audio, samplerate=samplerate)

        return data, mode, suffix, write_fn
    except (TypeError, RuntimeError):
        pass

    try:  # Hyperspectral
        data = np.load(filename).astype(float)
        mode = "hyperspectral"
        # suffix = ".png"
        # write_fn = _write_hyperspectral_as_img
        
        suffix = ".npy"
        write_fn = _write_hyperspectral

        return data, mode, suffix, write_fn
    except Exception as e:
        print(e)

    raise RuntimeError(f"No valid file input found for {filename}")


def shift(qc: QuantumCircuit, k: int = 1, targets=None, control=None):
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
        targets = qc.qubits

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

    tmp = list(enumerate(targets))
    for j, target in tmp if transpose else reversed(tmp):
        control = [qc.qubits[k] for k in targets[j + 1 :]]

        n_j = len(targets) - 1 - j
        i_max = 2 ** (n_j)

        theta_j = theta[:i_max, j]
        t_j = t[:i_max, j]
        phi_j = phi[:i_max, j]

        if t_j.any():
            t_j = -t_j
            qc.ucrz(t_j.tolist(), control, target)

        qc.ucry(theta_j.tolist(), control, target)

        if phi_j.any():
            qc.ucrz(phi_j.tolist(), control, target)


def quantum_convolution(
    data: np.array,
    fltr: np.array,
    noisy_execution: bool = True,
):
    npad = tuple((0, 2 ** int(np.ceil(np.log2(N))) - N) for N in fltr.shape)
    fltr = np.pad(fltr, pad_width=npad, mode="constant", constant_values=0)

    dims = data.shape
    psi = flatten_array(data, pad=True)
    psi, mag = normalize(psi, include_magnitude=True)

    dims_q = [to_qubits(x) for x in dims]

    num_qubits = sum(dims_q)
    num_states = 2**num_qubits

    fltr_shape_q = [int(np.ceil(np.log2(filter_size))) for filter_size in fltr.shape]
    num_ancilla = sum(fltr_shape_q)
    total_qubits = num_qubits + num_ancilla

    qc = QuantumCircuit(total_qubits)
    qc.initialize(psi, qc.qubits[:-num_ancilla])

    for i, (dim, fq) in enumerate(zip(dims_q[: fltr.ndim], fltr_shape_q)):
        filter_qubits = num_qubits + sum(fltr_shape_q[:i]) + np.arange(fq)
        qc.h(filter_qubits)

        # Shift operation
        qubits = list(sum(dims_q[:i]) + np.arange(dim))
        for i, control_qubit in enumerate(filter_qubits):
            shift(qc, -1, targets=qubits[i:], control=control_qubit)

    params, fltr_mag = normalize(fltr.flatten(order="F"), include_magnitude=True)

    roots = np.concatenate(
        [np.zeros(1, dtype=int), np.cumsum(dims_q[: fltr.ndim - 1])]
    )  # base
    filter_qubits = np.array(
        [root + np.arange(fq) for root, fq in zip(roots, fltr_shape_q)], dtype=int
    ).flatten()

    swap_targets = np.array(
        [
            num_qubits + sum(fltr_shape_q[:i]) + np.arange(fq)
            for i, fq in enumerate(fltr_shape_q)
        ],
        dtype=int,
    ).flatten()

    for a, b in zip(filter_qubits, swap_targets):
        qc.swap(a, b)

    c2q(qc, params, targets=swap_targets, transpose=True)

    ### Run

    backend = Aer.get_backend("aer_simulator")
    shots = backend.configuration().max_shots

    if noisy_execution:
        qc.measure_all()
    else:
        qc.save_statevector()

    job = execute(qc, backend=backend, shots=shots, seed_simulator=42069)

    result = job.result()

    if noisy_execution:
        counts = result.get_counts(qc)
        psi_out = from_counts(counts, shots=shots, num_qubits=total_qubits)
    else:
        psi_out = result.get_statevector(qc).data

    # dims = dims[:2]
    # num_states = np.prod(dims)

    ### Construct image
    i = 0
    data = psi_out.data[i * num_states : (i + 1) * num_states]
    norm = mag * fltr_mag * np.sqrt(2**num_ancilla)
    data = norm * data

    new_dims = [2 ** to_qubits(x) for x in dims]
    data = np.abs(data)
    data = data.reshape(new_dims, order="F")
    data = data[tuple([slice(s) for s in dims])]  # Remove padded zeroes

    return data


def _write_image(data: np.ndarray):
    # Convert data type
    data = data.astype(np.uint8)

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
    waveshow(data, sr=samplerate, color="#37b6f0")
    plt.axis("off")

    return partial(plt.savefig, dpi=300, bbox_inches="tight", pad_inches=0)


def _write_hyperspectral(data: np.ndarray):
    return partial(np.save, arr=data)


def _write_hyperspectral_as_img(data: np.ndarray):
    return partial(save_rgb, data=data)
