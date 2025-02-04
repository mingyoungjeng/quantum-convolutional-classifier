from __future__ import annotations
from typing import TYPE_CHECKING

import importlib
import tomllib
from pathlib import Path
from PIL import Image
from astropy.io import fits
from polars import DataFrame, read_csv
from matplotlib.figure import Figure
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from typing import Iterable, Callable, Any


def create_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def new_dir(dir: Path, overwrite=True) -> Path:
    if not isinstance(dir, Path):
        dir = Path(dir)

    if not overwrite:
        i = 1
        stem = dir.stem
        while dir.is_dir():
            dir = dir.with_name(f"{stem}_{i}")
            i += 1

    dir.mkdir(parents=True, exist_ok=overwrite)

    # Return path in case stem change needed
    return dir


def filename_labels(
    filename: Path, labels: str | Iterable[str] | None = None
) -> Path | Iterable[Path]:
    if labels is None:
        return tuple(
            filename.parent.absolute().glob(f"{filename.stem}_*{filename.suffix}")
        )

    if isinstance(labels, str):
        stem = f"{filename.stem}_{labels}"
        return filename.with_stem(stem)

    stems = (f"{filename.stem}_{label}" for label in labels)
    filenames = [filename.with_stem(stem) for stem in stems]
    return filenames


def save(filename: Path, fn: Callable[[Path], None], overwrite=True) -> None:
    if not isinstance(filename, Path):
        filename = Path(filename)

    create_parent(filename)

    if not overwrite:
        i = 1
        stem = filename.stem
        while filename.is_file():
            filename = filename.with_name(f"{stem}_{i}{filename.suffix}")
            i += 1

    fn(filename)

    # Return filename in case stem change needed
    return filename


def save_numpy_as_image(data: np.ndarray):
    # Convert data type
    data = np.abs(data).astype(np.uint8)

    # Create Image
    img = Image.fromarray(data)

    return img.save


def save_img(filename: Path, img: Image.Image, overwrite=True) -> None:
    return save(filename, img.save, overwrite=overwrite)


def save_fits(filename: Path, data: fits.HDUList, overwrite=True) -> None:
    if not isinstance(filename, Path):
        filename = Path(filename)
    filename = filename.with_suffix(".fits")

    return save(filename, data.writeto, overwrite=overwrite)


def save_dataframe_as_csv(filename: Path, df: DataFrame, overwrite=True) -> None:
    if not isinstance(filename, Path):
        filename = Path(filename)
    filename = filename.with_suffix(".csv")

    return save(filename, fn=df.write_csv, overwrite=overwrite)


def load_dataframe_from_csv(filename: Path, **kwargs) -> DataFrame | None:
    if not isinstance(filename, Path):
        filename = Path(filename)
    filename = filename.with_suffix(".csv")

    return read_csv(filename, **kwargs) if filename.is_file() else None


def draw(
    fig_ax: tuple[Figure, Axes],
    filename: Path | None = None,
    overwrite: bool = False,
    include_axis: bool = False,
) -> tuple[Figure, Axes] | Figure:
    """Wrapper for easier handling of matplotlib figures/axes"""
    fig, ax = fig_ax

    if filename is not None:
        save(filename, fig.savefig, overwrite=overwrite)

    return (fig, ax) if include_axis else fig


# def draw(func: Callable):
#     """Decorator for easier handling of matplotlib figures/axes"""

#     def wrapper(
#         *args,
#         filename: Path | None = None,
#         overwrite: bool = False,
#         include_axis: bool = False,
#         **kwargs,
#     ):
#         print(func(args, kwargs))
#         return _draw(func(args, kwargs), filename, overwrite, include_axis)

#     return wrapper


def lookup(arg: str, root: str = None):
    root, module = arg.rsplit(".", 1) if "." in arg else (root, arg)

    if root is None:
        root = __name__

    if isinstance(root, str):
        root = importlib.import_module(root)

    return getattr(root, module, arg)


def load_toml(filename: Path) -> dict[str, Any]:
    with open(filename, mode="rb") as file:
        return tomllib.load(file)
