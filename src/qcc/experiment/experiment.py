from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from pathlib import Path
from copy import copy, deepcopy
from multiprocessing import Pool, set_start_method

from attrs import define, field
import polars as pl
import matplotlib.pyplot as plt
from qcc.file import (
    draw,
    filename_labels,
    save_dataframe_as_csv as save,
    load_dataframe_from_csv as load,
)
from qcc.experiment.logger import Logger

if TYPE_CHECKING:
    from typing import Optional, Mapping, Any
    from qcc.experiment.logger import SchemaDefinition

    # TODO: Add type for class that has a Logger


@define
class Experiment:
    """Perform and aggregate multiple experimental trials"""

    cls: Optional[Any] = field(default=None)
    num_trials: int = 1
    results_schema: Optional[SchemaDefinition] = None
    dfs: dict[str, pl.DataFrame] = field(repr=False, factory=dict)

    @cls.validator
    def _check_if_logger(self, _, value):
        # Do nothing if cls is not set
        if value is None:
            return

        # Check if has attribute logger
        if hasattr(value, "logger"):
            # Check if logger is correct instance
            if not isinstance(value.logger, Logger):
                raise TypeError("Logger bad")
        else:
            raise AttributeError("No Logger")

    @property
    def metrics(self):
        return self.dfs.keys()

    def _run_trial(
        self,
        idx: int | Iterable[int],
        cls: Optional[Any] = None,
        rename: bool = False,
    ) -> Mapping[str, pl.DataFrame]:
        if cls is None:
            cls = copy(self.cls)

        # Setup logging
        if rename:
            i = max(idx) if isinstance(idx, Iterable) else idx
            cls.logger = Logger.copy(cls.logger, name=f"{cls.logger.name}_trial_{i}")

        # Perform trial and get results
        results_df = pl.DataFrame([cls()], schema=self.results_schema)

        # Combine DataFrames
        if rename:
            self._make_unique_columns(cls.logger.dfs, idx)
        cls.logger.dfs["results"] = results_df

        return cls.logger.dfs

    @staticmethod
    def _make_unique_columns(
        dfs: Mapping[str, pl.DataFrame],
        idx: int | Iterable[int],
    ) -> None:
        for i, key in enumerate(dfs):
            if key == "results":
                continue

            i = idx[i] if isinstance(idx, Iterable) else idx
            col = pl.col(key).suffix(f"_{i}")
            dfs[key] = dfs.get(key).select(col)

    def _merge_dfs(
        self, dfs: Mapping[str, pl.DataFrame], filename: Optional[Path] = None
    ) -> None:
        for metric, df in dfs.items():
            if metric not in self.dfs:
                self.dfs[metric] = df
                continue

            if metric == "results":  # results DataFrame
                self.dfs[metric].vstack(df, in_place=True)
            else:
                self.dfs[metric].hstack(df, in_place=True)

            # Save DataFrames
            self.save(filename, overwrite=True)

    def read(self, filename: Optional[Path] = None) -> None:
        if filename is None:
            return
        else:
            filename = filename.with_suffix(".csv")

        # TODO: refactor this section
        metrics = None if self.cls is None else self.cls.logger.dfs.keys()
        filenames: Iterable[Path] = filename_labels(filename, metrics)
        if metrics is None:
            metrics = (f.stem.removeprefix(f"{filename.stem}_") for f in filenames)

        dfs = [load(f) for f in filenames]
        dfs = dict((m, df) for m, df in zip(metrics, dfs) if df is not None)
        self._merge_dfs(dfs)

    def save(self, filename: Optional[Path] = None, overwrite: bool = False) -> None:
        if filename is None:
            return

        filenames = filename_labels(filename, self.metrics)
        for f, df in zip(filenames, self.dfs.values()):
            save(f, df, overwrite=overwrite)

    def __call__(
        self,
        *,
        filename: Optional[Path] = None,
        parallel: bool = False,
    ) -> pl.DataFrame:
        # Import from path
        self.read(filename)

        offset = {len(df.columns) for df in self.dfs.values()}
        offset = max(offset) if len(offset) > 0 else 0
        if parallel:  # TODO: doesn't work
            try:  # For CUDA compatibility in PyTorch
                set_start_method("spawn")
            except RuntimeError:
                pass

            with Pool() as pool:
                args = (i + offset for i in range(self.num_trials))
                results = pool.imap(self._run_trial, args, rename=self.num_trials > 1)
                for dfs in results:
                    self._merge_dfs(dfs, filename)
        else:
            for i in range(self.num_trials):
                dfs = self._run_trial(i + offset, rename=self.num_trials > 1)
                self._merge_dfs(dfs, filename)

        return self.dfs.get("results")

    @staticmethod
    def aggregate(name: str, op: str):
        regex = f"^{name}(_[0-9]+)?$"
        fn = getattr(pl.element(), op)
        expr = pl.concat_list(pl.col(regex)).list.eval(fn()).list.first()

        return expr.alias(f"{name}_{op}")

    def draw(self, filename: Path = None, include_axis: bool = False):
        subplots = []
        for metric, df in self.dfs.items():
            if metric == "results":
                continue

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

        if filename is None:
            filenames = (None for _ in self.metrics)
        else:
            filenames = filename_labels(filename.with_suffix(".png"), self.metrics)

        return tuple(
            draw((fig, ax), f, overwrite=False, include_axis=include_axis)
            for (fig, ax), f in zip(subplots, filenames)
        )
