from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path

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
