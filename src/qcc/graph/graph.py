from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path

import polars as pl
import plotly.graph_objects as go
from qcc.file import load_toml

from qcc.experiment import Experiment

if TYPE_CHECKING:
    from typing import Optional

CWD = Path(__file__).parent
graph_template = go.layout.Template(load_toml(CWD / "supercomputing23.toml"))


def plot(
    *args,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: Optional[tuple[int, int]] = None,
    **kwargs,
) -> go.Figure:
    layout = dict()
    if title is not None:
        layout["title_text"] = "<b>" + title

        num_br = title.count("<br>")
        layout["margin_t"] = 40 + (30 * num_br)
        kwargs["title_y"] = kwargs.get("title_y", 1) - (0.06 * num_br)
    else:
        layout["title_text"] = ""
    if xlabel is not None:
        layout["xaxis_title_text"] = "<b>" + xlabel

        num_br = xlabel.count("<br>")
        layout["margin_b"] = 34 * (num_br + 2)
    if ylabel is not None:
        layout["yaxis_title_text"] = "<b>" + ylabel

        layout["margin_l"] = 84
        layout["title_x"] = (42 + ((600 - 42) / 2)) / 600
    if legend is not None:
        layout["legend_x"] = legend[0]
        layout["legend_y"] = legend[1]
    layout.update(kwargs)

    for arg in args:
        if "name" in arg:
            arg["name"] = "<b>" + arg.get("name")

    fig = go.Figure(data=args, layout=layout)

    return fig


def process_dataframe(df: pl.DataFrame, as_plotly: bool = False):
    x = df[""] if "" in df.columns else list(range(df.select(pl.count()).item()))

    exprs = (Experiment.aggregate(op=op) for op in ["median", "mean", "std"])
    df = df.select(*exprs)

    exprs = (
        getattr(pl.col("mean"), op)(pl.col("std"))
        .sub(pl.col("median"))
        .abs()
        .alias(name)
        for name, op in zip(["error_y+", "error_y-"], ["add", "sub"])
    )

    df = df.select(pl.col("median").alias("y"), *exprs)

    if not as_plotly:
        return df

    y = list(df["y"])
    error_y = dict(
        type="data",
        symmetric=False,
        array=list(df["error_y+"]),
        arrayminus=list(df["error_y-"]),
    )
    return dict(x=x, y=y, error_y=error_y)
