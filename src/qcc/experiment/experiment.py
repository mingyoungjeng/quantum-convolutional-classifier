from __future__ import annotations
from typing import TYPE_CHECKING, Any

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
    fn: Callable = field()
    num_trials: int = 1

    @fn.default
    def _default_fn(self):
        return getattr(self.cls, "__call__", lambda: None)

    results_schema: SchemaDefinition = None
    dfs: list[Optional[pl.DataFrame]] = field(init=None, factory=lambda: [None, None])
    metrics: list[str] = field(init=None, factory=list)

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

        if filename is not None:  # ideal output filenames
            filenames = filename, filename.with_stem(f"{filename.stem}_results")

            if merge:
                self.dfs = [load(f) for f in filenames]
                offset = len(self.dfs[0].columns) if self.dfs[0] is not None else 0
            else:  # Reserve file names
                filenames = [save(f, pl.DataFrame(), False) for f in filenames]

        for i in range(self.num_trials):
            if filename and merge:
                i += offset

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
            idf = idf.select(pl.col(self.metrics).suffix(f"_{i}"))

            if self.dfs[0] is None:
                self.dfs[0] = idf
            else:
                self.dfs[0].hstack(idf, in_place=True)

            if self.dfs[1] is None:
                self.dfs[1] = results_row
            else:
                self.dfs[1].vstack(results_row, in_place=True)

            if filename is not None:
                for f, df in zip(filenames, self.dfs):
                    save(f, df, overwrite=True)

        self.cls.logger = logger
        return self.dfs[1]

    @staticmethod
    def aggregate(name: str, op: str):
        regex = f"^{name}_[0-9]+$"
        fn = getattr(pl.element(), op)
        expr = pl.concat_list(pl.col(regex)).list.eval(fn()).list.first()

        return expr.alias(f"{name}_{op}")

    def draw(self, filename=None, include_axis: bool = False):
        # Aggregate columns
        exprs = product(self.metrics, ["mean", "std"])
        exprs = tuple(self.aggregate(*expr) for expr in exprs)
        df = self.dfs[0].with_columns(*exprs)  # df.select(*exprs)

        subplots = []
        for metric in self.metrics:
            fig, ax = plt.subplots()
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

    def pass_args(self, *args, **kwargs):
        self.fn = lambda: self.cls(*args, **kwargs)
