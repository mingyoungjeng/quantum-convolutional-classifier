from __future__ import annotations
from typing import TYPE_CHECKING, Any

from pathlib import Path
from itertools import product
from attrs import define, field
import polars as pl
import matplotlib.pyplot as plt
from qcc.file import draw, save_dataframe_as_csv
from qcc.experiment.logger import Logger

if TYPE_CHECKING:
    from typing import Optional, Callable
    from qcc.experiment.logger import SchemaDefinition


@define
class Experiment:
    """Perform and aggregate multiple experimental trials"""

    cls: Any = field()
    num_trials: int = 1

    results_schema: SchemaDefinition = None
    df: pl.DataFrame = field(init=None, default=None)
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
        self, fn: Optional[Callable] = None, filename: Path = None, merge: bool = False
    ):
        if fn is None:
            fn = self.cls.__call__

        logger_name = self.cls.logger.name
        self.metrics = self.cls.logger.df.columns[1:]

        results = pl.DataFrame(schema=self.results_schema)

        if filename is not None:
            # Reserve file name
            filename = save_dataframe_as_csv(filename, pl.DataFrame(), overwrite=False)
            results_filename = filename.with_stem(f"{filename.stem}_results")
            results_filename = save_dataframe_as_csv(results_filename, results, False)

        for i in range(self.num_trials):
            # Setup DataFrame
            idf = pl.DataFrame(schema=self.cls.logger.df.schema)

            # Setup logging
            self.cls.logger = Logger(
                df=idf,
                name=f"{logger_name}_trial_{i}",
                format=self.cls.logger.format,
            )

            # Perform trial
            results_row = pl.DataFrame([fn()], schema=self.results_schema)

            # Combine DataFrames
            idf = idf.select(pl.col(self.metrics).suffix(f"_{i}"))
            if i == 0:
                df = idf
                results = results_row
            else:
                df.hstack(idf, in_place=True)
                results.vstack(results_row, in_place=True)

            if filename is not None:
                save_dataframe_as_csv(filename, df, overwrite=True)
                save_dataframe_as_csv(results_filename, results, overwrite=True)

        # Aggregate columns
        exprs = product(self.metrics, ["mean", "std"])
        exprs = tuple(self.aggregate(*expr) for expr in exprs)
        self.df = df.with_columns(*exprs)  # df.select(*exprs)

        if filename is not None:
            save_dataframe_as_csv(filename, self.df, overwrite=True)
            # save_dataframe_as_csv(results_filename, results, overwrite=True)

        return results

    @staticmethod
    def aggregate(name: str, op: str):
        regex = f"^{name}_[0-9]+$"
        fn = getattr(pl.element(), op)
        expr = pl.concat_list(pl.col(regex)).list.eval(fn()).list.first()

        return expr.alias(f"{name}_{op}")

    def draw(self, filename=None, include_axis: bool = False):
        subplots = []
        for metric in self.metrics:
            fig, ax = plt.subplots()
            mean = self.df.get_column(f"{metric}_mean").to_numpy()
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

    def callable_wrapper(self, *args, **kwargs):
        return lambda: self.cls(*args, **kwargs)
