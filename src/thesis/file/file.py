from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from PIL import Image
from astropy.io import fits
from polars import DataFrame

if TYPE_CHECKING:
    from typing import Callable


def create_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def save(filename: Path, fn: Callable[[Path], None], overwrite=True):
    if not isinstance(filename, Path):
        filename = Path(filename)

    create_parent(filename)

    if not overwrite:
        i = 1
        while filename.is_file():
            filename = filename.with_name(f"{filename.stem}_{i}{filename.suffix}")
            i += 1

    return fn(filename)


def save_img(filename: Path, img: Image.Image, overwrite=True):
    save(filename, img.save, overwrite=overwrite)


def save_fits(filename: Path, data: fits.HDUList, overwrite=True):
    filename = filename.with_suffix(".fits")
    save(filename, data.writeto, overwrite=overwrite)


def save_dataframe_as_csv(filename: Path, df: DataFrame, overwrite=True):
    filename = filename.with_suffix(".csv")

    save(filename, fn=df.write_csv, overwrite=overwrite)
