from __future__ import annotations
from typing import TYPE_CHECKING

import logging
from pathlib import Path
from datetime import datetime
from attrs import define, field
import polars as pl
from thesis.file import save_dataframe_as_csv

if TYPE_CHECKING:
    from typing import Optional

    SchemaDefinition = list[tuple[str, pl.DataType]]


@define(frozen=True)
class Logger:
    df: pl.DataFrame = field(repr=False)
    name: str = field(default=__name__)
    format: str = None

    @df.validator
    def _check_df(self, _, df: pl.DataFrame):
        if pl.Datetime not in df.dtypes:
            df.insert_at_idx(0, pl.Series("time", dtype=pl.Datetime))

    @name.validator
    def _setup_logging(self, _, name):
        if name is not None:
            self.logger.handlers.clear()
            self.logger.propagate = False

            self.logger.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            if self.format is not None:
                formatter = logging.Formatter(self.format)
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

    def info(self, msg: str, silent: bool = False) -> None:
        if silent:
            return

        self.logger.info(msg)

    def save(self, filename: Optional[Path] = None, overwrite=True) -> None:
        if filename is None:
            filename = Path(f"{self.name}.csv")

        save_dataframe_as_csv(filename, self.df, overwrite=overwrite)

    @classmethod
    def from_schema(cls, schema: SchemaDefinition, *args, **kwargs) -> Logger:
        return cls(pl.DataFrame(schema=schema), *args, **kwargs)

    @classmethod
    def copy(cls, logger: Logger):
        df = pl.DataFrame(schema=logger.df.schema)
        name = logger.name
        fmt = logger.format

        return cls(df=df, name=name, format=fmt)
