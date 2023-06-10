from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from PIL import Image
from astropy.io import fits
from polars import DataFrame
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from typing import Callable, Optional


def create_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


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

    return fn(filename)


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


def draw(
    fig_ax: tuple[Figure, Axes],
    filename: Optional[Path] = None,
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
#         filename: Optional[Path] = None,
#         overwrite: bool = False,
#         include_axis: bool = False,
#         **kwargs,
#     ):
#         print(func(args, kwargs))
#         return _draw(func(args, kwargs), filename, overwrite, include_axis)

#     return wrapper
