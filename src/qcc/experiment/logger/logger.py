from __future__ import annotations
from typing import TYPE_CHECKING

import logging
from pathlib import Path
from datetime import datetime
from attrs import define, field
import polars as pl
from qcc.file import save_dataframe_as_csv

if TYPE_CHECKING:
    from typing import Optional, Mapping

    SchemaDefinition = list[tuple[str, pl.DataType]]


@define(frozen=True)
class Logger:
    """Logs important values in DataFrame(s)"""

    dfs: Mapping[str, pl.DataFrame] = field(repr=False)
    name: str = field(default=__name__)
    format: str = None

    def _check_df(self, _, df: pl.DataFrame):
        if pl.Datetime not in df.dtypes:
            df.insert_at_idx(0, pl.Series("time", dtype=pl.Datetime))

    @dfs.validator
    def _check_dfs(self, _, dfs: Mapping[str, pl.DataFrame]):
        for df in dfs.values():
            self._check_df(_, df)

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

    def log(self, /, silent: bool = False, **kwargs):
        now = datetime.now()

        for key, value in kwargs.items():
            if key in self.dfs:
                df = self.dfs.get(key)
                row = pl.DataFrame([(now, value)], schema=df.schema)
                df.extend(row)
            else:
                self.logger.warning(f"Key not found: {key}")

                df = pl.DataFrame({"time": [now], key: [value]})
                self.dfs[key] = df

        # msg = row.select(row.columns[1:]).to_dicts()[0]
        self.info(str(kwargs), silent=silent)

    def info(self, msg: str, silent: bool = False) -> None:
        if silent:
            return

        self.logger.info(msg)

    def save(self, filename: Optional[Path] = None, overwrite=True) -> None:
        if filename is None:
            filename = Path(f"{self.name}.csv")

        for key, df in self.dfs.items():
            key = filename.with_stem(f"{filename.stem}_{key}")
            save_dataframe_as_csv(key, df, overwrite=overwrite)

    @classmethod
    def from_schema(cls, schema: SchemaDefinition, *args, **kwargs) -> Logger:
        dfs = dict((s[0], pl.DataFrame(schema=[s])) for s in schema)
        return cls(dfs, *args, **kwargs)

    @classmethod
    def copy(
        cls,
        logger: Logger,
        /,
        dfs: Mapping[str, pl.DataFrame] = None,
        name: str = None,
        fmt: str = None,
    ):
        if dfs is None:
            dfs = dict(
                (key, pl.DataFrame(schema=df.schema))
                for (key, df) in logger.dfs.items()
            )

        if name is None:
            name = logger.name

        if fmt is None:
            fmt = logger.format

        return cls(dfs=dfs, name=name, format=fmt)
