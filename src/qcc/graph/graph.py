from __future__ import annotations
from typing import TYPE_CHECKING

import polars as pl
import plotly.graph_objects as go

from qcc.experiment import Experiment

if TYPE_CHECKING:
    from typing import Optional

def plot(
    *args,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend: Optional[tuple[int, int]] = None,
    **kwargs,
):
    layout = dict()
    if title is not None:
        layout["title_text"] = title
        
        num_br = title.count("<br>")
        layout["margin_t"] = 40 + (52 * num_br)
        kwargs["title_y"] = kwargs.get("title_y", 1) - (0.08 * num_br)
    else:
        layout["title_text"] = ""
        layout["margin_t"] = 40
    if xlabel is not None:
        layout["xaxis_title_text"] = xlabel
        
        num_br = xlabel.count("<br>")
        layout["margin_b"] = 40 * (num_br + 2)
    if ylabel is not None:
        layout["yaxis_title_text"] = ylabel
        
        num_br = ylabel.count("<br>")
        layout["margin_l"] = 44 + (52 * (num_br + 1))
        # layout["title_x"] = (64 + ((600 - 64)/2)) / 600
    if legend is not None:
        layout["legend_x"] = legend[0]
        layout["legend_y"] = legend[1]
    layout.update(kwargs)
    
    fig = go.Figure(
        data=args,
        layout=layout
    )
    
    return fig

def process_dataframe(df: pl.DataFrame, as_plotly: bool = False):
    exprs = (Experiment.aggregate(op=op) for op in ["median", "mean", "std"])
    df = df.select(*exprs)

    exprs = (getattr(pl.col("mean"), op)(pl.col("std")).sub(pl.col("median")).abs().alias(name) 
            for name, op in zip(["error_y+", "error_y-"], ["add", "sub"]))

    df = df.select(pl.col("median").alias("y"), *exprs)
    
    if not as_plotly:
        return df
    
    y = list(df["y"])
    error_y = dict(
        type='data',
        symmetric=False,
        array=list(df["error_y+"]),
        arrayminus=list(df["error_y-"])
    )
    return dict(y=y, error_y=error_y)
