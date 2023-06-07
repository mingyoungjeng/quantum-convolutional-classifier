from __future__ import annotations
from typing import TYPE_CHECKING, Any

from itertools import product
from attrs import define, field
import polars as pl
import matplotlib.pyplot as plt
from qcnn.experiment.logger import Logger

if TYPE_CHECKING:
    from typing import Optional, Callable
    from qcnn.experiment.logger import SchemaDefinition


@define
class Experiment:
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

    def __call__(self, *args, fn: Optional[Callable] = None, **kwargs):
        if fn is None:
            fn = self.cls.__call__

        logger_name = self.cls.logger.name
        self.metrics = self.cls.logger.df.columns[1:]

        results = []
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
            results += [fn(*args, **kwargs)]

            # Combine DataFrames
            idf = idf.select(pl.col(self.metrics).suffix(f"_{i}"))
            df = idf if i == 0 else df.hstack(idf)

        # Aggregate columns
        exprs = product(self.metrics, ["mean", "std"])
        exprs = tuple(self.aggregate(*expr) for expr in exprs)
        self.df = df.with_columns(*exprs)  # df.select(*exprs)

        return pl.DataFrame(results, schema=self.results_schema)

    @staticmethod
    def aggregate(name: str, op: str):
        regex = f"^{name}_[0-9]+$"
        fn = getattr(pl.element(), op)
        expr = pl.concat_list(pl.col(regex)).list.eval(fn()).list.first()

        return expr.alias(f"{name}_{op}")

    def draw(self, include_axis: bool = False):
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

        figs, axes = zip(*subplots)
        return figs, axes if include_axis else figs
