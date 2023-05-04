from typing import Union
from pathlib import Path
from PIL import Image
from astropy.io import fits


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
