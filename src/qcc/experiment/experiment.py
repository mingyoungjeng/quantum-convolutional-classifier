from __future__ import annotations
from typing import TYPE_CHECKING, Any

from functools import partial as _partial
from pathlib import Path
from itertools import product

from attrs import define, field
import polars as pl
import matplotlib.pyplot as plt
from qcc.file import (
    draw,
    save_dataframe_as_csv as save,
    load_dataframe_from_csv as load,
)
from qcc.experiment.logger import Logger

if TYPE_CHECKING:
    from typing import Optional, Callable
    from qcc.experiment.logger import SchemaDefinition


@define
class Experiment:
    """Perform and aggregate multiple experimental trials"""

    cls: Any = field()
    num_trials: int = 1
    fn: Callable = field(kw_only=True)

    results_schema: Optional[SchemaDefinition] = None
    dfs: list[Optional[pl.DataFrame]] = field(init=None, factory=lambda: [None])
    metrics: list[str] = field(init=None, factory=list)

    @fn.default
    def _default_fn(self):
        return getattr(self.cls, "__call__", lambda: None)

    @cls.validator
    def _check_if_logger(self, _, value):
        # Check if has attribute logger
        if hasattr(value, "logger"):
            # Check if logger is correct instance
            if not isinstance(value.logger, Logger):
                raise TypeError("Logger bad")
        else:
            raise AttributeError("No Logger")

    def __call__(
        self,
        fn: Optional[Callable] = None,
        *,
        filename: Optional[Path] = None,
        merge: bool = True,
    ):
        if fn is None:
            fn = self.fn

        logger = self.cls.logger
        self.metrics = logger.df.columns[1:]
        self.dfs = [None for _ in range(len(self.metrics) + 1)]

        if filename is not None:  # ideal output filenames
            filenames = "results", *self.metrics
            filenames = [filename.with_stem(f"{filename.stem}_{f}") for f in filenames]

            if merge:
                self.dfs = [load(f) for f in filenames]
            else:  # Reserve file names
                filenames = [save(f, pl.DataFrame(), False) for f in filenames]

        offsets = tuple(0 if df is None else len(df.columns) for df in self.dfs)
        for i in range(self.num_trials):
            idx = (i + offset for offset in offsets)

            # Setup DataFrame
            idf = pl.DataFrame(schema=logger.df.schema)

            # Setup logging
            self.cls.logger = Logger(
                df=idf,
                name=f"{logger.name}_trial_{i}",
                format=logger.format,
            )

            # Perform trial
            results_row = pl.DataFrame([fn()], schema=self.results_schema)

            # Combine DataFrames
            idfs = [
                idf.select(pl.col(m).suffix(f"_{j}")) for j, m in zip(idx, self.metrics)
            ]

            for j, idf in enumerate((results_row, *idfs)):
                if self.dfs[j] is None:
                    self.dfs[j] = idf
                    continue

                if j == 0:
                    self.dfs[j].vstack(idf, in_place=True)
                else:
                    self.dfs[j].hstack(idf, in_place=True)

            if filename is not None:
                for f, df in zip(filenames, self.dfs):
                    save(f, df, overwrite=True)

        self.cls.logger = logger
        return self.dfs[0]

    @staticmethod
    def aggregate(name: str, op: str):
        regex = f"^{name}_[0-9]+$"
        fn = getattr(pl.element(), op)
        expr = pl.concat_list(pl.col(regex)).list.eval(fn()).list.first()

        return expr.alias(f"{name}_{op}")

    def draw(self, filename=None, include_axis: bool = False):

        subplots = []
        for df, metric in zip(self.dfs[1:], self.metrics):
            fig, ax = plt.subplots()
            
            # Aggregate columns
            exprs = ["mean", "std"]
            exprs = tuple(self.aggregate(metric, expr) for expr in exprs)
            df = df.with_columns(*exprs)  # df.select(*exprs)
            
            mean = df.get_column(f"{metric}_mean").to_numpy()
            # std = self.df.get_column(f"{metric}_std").to_numpy()
            ax.plot(mean)
            # ax.errorbar(x=range(len(mean)), y=mean, yerr=std)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(metric.capitalize())
            subplots += [(fig, ax)]

        return tuple(
            draw((fig, ax), filename, overwrite=False, include_axis=include_axis)
            for (fig, ax) in subplots
        )

    def partial(self, *args, **kwargs):
        self.fn = _partial(self.fn, *args, **kwargs)
