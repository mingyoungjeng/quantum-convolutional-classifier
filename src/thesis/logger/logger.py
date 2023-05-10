from __future__ import annotations
from typing import TYPE_CHECKING

import logging
from pathlib import Path
from datetime import datetime
from attrs import define, field
import polars as pl

if TYPE_CHECKING:
    from typing import Optional

    SchemaDefinition = list[tuple[str, pl.DataType]]


@define(frozen=True)
class Logger:
    df: pl.DataFrame = field(repr=False)

    @df.validator
    def _check_df(self, _, df: pl.DataFrame):
        if pl.Datetime not in df.dtypes:
            df.insert_at_idx(0, pl.Series("time", dtype=pl.Datetime))

    name: str = field(default=__name__)

    @name.validator
    def _setup_logging(self, _, __):
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)

        self.logger.addHandler(ch)

    @property
    def logger(self):
        return logging.getLogger(self.name)

    def log(self, *args, silent: bool = False):
        now = datetime.now()
        row = pl.DataFrame([(now,) + args], schema=self.df.schema)
        self.df.extend(row)

        if not silent:
            msg = row.select(row.columns[1:]).to_dicts()[0]
            self.logger.info(msg)

    def save(self, filename: Optional[Path] = None, overwrite=True):
        if filename is None:
            filename = Path(f"{self.name}.csv")
        if not isinstance(filename, Path):
            filename = Path(filename)

        filename.parent.mkdir(parents=True, exist_ok=True)

        if not overwrite:
            i = 1
            while filename.is_file():
                filename = filename.with_name(f"{filename.stem}_{i}{filename.suffix}")
                i += 1

        self.df.write_csv(filename)

    @classmethod
    def from_schema(cls, schema: SchemaDefinition, *args, **kwargs) -> Logger:
        return cls(pl.DataFrame(schema=schema), *args, **kwargs)
