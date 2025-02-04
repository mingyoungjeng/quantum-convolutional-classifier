"""
Command line interface

Very useful for rapid and efficient experiment execution (e.g. on cluster)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import re
from ast import literal_eval
from fnmatch import fnmatch
from pathlib import Path
import logging
from multiprocessing import Pool, set_start_method

import click

from .ml import Classify
from .pooling import DimensionReduction, _pooling
from .convolution import Filters, _convolution
from qcc.experiment import Experiment

if TYPE_CHECKING:
    from typing import Iterable

log = logging.getLogger(__name__)


# ==== boilerplate ==== #
class ObjectType(click.ParamType):
    """
    click.ParamType for compatibility with TOML config files
    """

    name = "object"


class DictType(click.ParamType):
    """
    click.ParamType for compatibility with TOML config files
    """

    name = "key:value"

    def convert(self, value, param, ctx):
        try:
            key, value = re.split(r":|=", value)

            # Try to evaluate value
            try:
                value = literal_eval(value)
            except ValueError:
                pass

            return key, value
        except ValueError:
            self.fail(f"{value!r} is not a valid {self.name}", param, ctx)


def create_results(ctx, param, value: Path):
    results_dir = value / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


@click.group()
@click.pass_context
def cli(ctx: click.Context):
    """TODO: put in options for log levels"""

    # Set context settings
    ctx.show_default = True


# ==== classify command ==== #


@cli.command()
@click.pass_context
@click.option(
    "-n",
    "--name",
    required=True,
    type=str,
    help="Identifier of the experiment",
)
@click.option(
    "--ansatz",
    type=ObjectType(),
)
@click.option(
    "-d",
    "--dimensions",
    type=str,
)
@click.option(
    "-l",
    "--num_layers",
    default=0,
    type=int,
)
@click.option(
    "--dataset",
    required=True,
    default="MNIST",
    type=ObjectType(),
)
@click.option(
    "-c",
    "--classes",
    default=(0, 1),
    type=str,
)
@click.option(
    "--transform",
    type=ObjectType(),
)
@click.option(
    "--optimizer",
    required=True,
    default="Adam",
    type=ObjectType(),
)
@click.option(
    "-e",
    "--epoch",
    default=1,
    type=int,
)
@click.option(
    "-b",
    "--batch_size",
    default=1,
    type=str,
)
@click.option(
    "--loss",
    "--cost",
    required=True,
    default="CrossEntropyLoss",
    type=ObjectType(),
)
@click.option(
    "-t",
    "--num_trials",
    default=1,
    type=int,
)
@click.option(
    "-o",
    "--output_dir",
    required=True,
    type=click.Path(
        exists=True,
        path_type=Path,
        resolve_path=True,
        file_okay=False,
        writable=True,
    ),
    default=Path.cwd(),
    show_default=False,
    callback=create_results,
    help="Output directory",
)
@click.option(
    "-ao",
    "--module_options",
    type=DictType(),
    multiple=True,
    callback=lambda *x: dict(x[2]),
)
@click.option(
    "-oo",
    "--optimizer_options",
    type=DictType(),
    multiple=True,
    callback=lambda *x: dict(x[2]),
)
@click.option(
    "--quantum/--classical",
    default=True,
    required=True,
)
@click.option(
    "--verbose",
    is_flag=True,
)
def classify(**kwargs):
    """Quantum and classical classification / supervised learning"""
    _classify(**kwargs)


def _setup_module(root: str, obj: str = None):
    """String formatting of Python modules for string => module conversion down the line"""
    if obj is None or "." in obj:
        return obj

    return f"{root}.{obj}"


def _classify(
    ansatz: str,
    dataset: str,
    optimizer: str,
    loss: str,
    transform: str,
    *,
    quantum: bool,
    **kwargs,
):
    """See Classify for more details"""

    kwargs["ansatz"] = _setup_module("qcc.quantum.pennylane.ansatz", ansatz)
    kwargs["dataset"] = _setup_module("torchvision.datasets", dataset)
    kwargs["optimizer"] = _setup_module("torch.optim", optimizer)
    kwargs["loss"] = _setup_module("torch.nn", loss)
    kwargs["transform"] = _setup_module("qcc.ml.data", transform)
    kwargs["is_quantum"] = quantum

    cmd = Classify(**kwargs)
    cmd()


# ==== load command ==== #


@cli.command()
@click.pass_context
@click.argument(
    "paths",
    nargs=-1,
    type=click.Path(
        exists=True,
        path_type=Path,
        resolve_path=True,
    ),
)
@click.option(
    "-g",
    "--glob",
    required=True,
    type=str,
    default=r"**/*.toml",
    help="Config file glob pattern",
)
@click.option(
    "-p",
    "--parallel",
    is_flag=True,
    help="Execute all tests in parallel",
)
@click.option(
    "-o",
    "--output_dir",
    required=True,
    type=click.Path(
        exists=True,
        path_type=Path,
        resolve_path=True,
        file_okay=False,
        writable=True,
    ),
    default=Path.cwd(),
    show_default=False,
    callback=create_results,
    help="Output directory",
)
def load(
    ctx,
    paths: Iterable[Path],
    glob: str,
    parallel: bool,
    output_dir: Path,
) -> None:
    """Load files(s) and perform classification"""

    toml_files = set()
    for path in paths:
        if path.is_dir():
            toml_files.update(path.glob(glob))
        else:
            if fnmatch(path, glob):
                toml_files.add(path)

    cmds = (Classify.from_toml(toml, output_dir=output_dir) for toml in toml_files)

    # parallel execution of files(s) in threadpool
    if parallel:
        return _classify_pool(cmds)

    # ==== serial execution of files(s) ==== #
    errs = []
    for cmd in cmds:
        try:
            cmd()
        except Exception as e:
            errs.append(e)

    if len(errs) > 0:
        for error_message in errs:
            log.error(error_message)
        raise RuntimeError(f"{len(errs)} file(s) encountered an error")

    return


def _classify_pool(cmds: Iterable[Classify]):
    experiments: dict[str, Experiment] = dict()
    output_dir: Path = None

    try:  # For CUDA compatibility in PyTorch
        set_start_method("spawn")
    except RuntimeError as e:
        log.warning(e)

    with Pool() as pool:
        # ==== generate experiments ==== #
        args = []
        for cmd in cmds:
            num_trials = cmd.num_trials
            if output_dir is None:
                output_dir = cmd.output_dir
            filename = output_dir / cmd.name / cmd.name

            cmd.num_trials = 1
            cmd.output_dir = None

            experiments[cmd.name] = Experiment()
            experiments[cmd.name].read(filename)

            args += (cmd for _ in range(num_trials))

        # ==== run experiments in thread pool ==== #
        results = pool.imap_unordered(Classify.__call__, args)

        # ==== write results to dataframe ==== #
        for name, dfs in results:
            experiment = experiments[name]

            offset = {
                len(df.columns)
                for key, df in experiment.dfs.items()
                if key != "results"
            }
            offset = max(offset) if len(offset) > 0 else 0

            experiment._make_unique_columns(dfs, offset)
            experiment._merge_dfs(dfs)

            filename = output_dir / name / name
            experiment.save(filename, overwrite=True)
            experiment.draw(filename, overwrite=True, close=True)


# ==== TODO: convolution command ==== #

# ==== pooling command ==== #


@cli.command()
@click.pass_context
@click.option(
    "--method",
    "decomposition_type",
    type=click.Choice(
        [e.value for e in DimensionReduction],
        case_sensitive=False,
    ),
)
@click.option(
    "-l",
    "--decomposition_levels",
    type=int,
    multiple=True,
    default=[0],
)
@click.option(
    "-i",
    "--inputs",
    required=True,
    type=click.Path(
        exists=True,
        path_type=Path,
        resolve_path=True,
        file_okay=False,
        writable=True,
    ),
    default=Path.cwd() / "data",
    show_default=False,
    help="Input data",
)
@click.option(
    "-o",
    "--output_dir",
    required=True,
    type=click.Path(
        exists=True,
        path_type=Path,
        resolve_path=True,
        file_okay=False,
        writable=True,
    ),
    default=Path.cwd(),
    show_default=False,
    callback=create_results,
    help="Output directory",
)
@click.option(
    "--noise_free/--noisy",
    default=True,
    required=True,
)
def pooling(ctx, **kwargs):
    """Multilevel, multidimensional pooling / dimension reduction / downsampling"""
    _pooling(**kwargs)


# ==== convolution ==== #
@cli.command()
@click.pass_context
@click.option(
    "--filter",
    "filter_name",
    type=click.Choice(
        [e.value for e in Filters],
        case_sensitive=False,
    ),
)
@click.option(
    "-s",
    "--shape",
    "filter_dims",
    type=int,
    multiple=True,
    default=[3, 3],
)
@click.option(
    "-i",
    "--inputs",
    required=True,
    type=click.Path(
        exists=True,
        path_type=Path,
        resolve_path=True,
        file_okay=False,
        writable=True,
    ),
    default=Path.cwd() / "data",
    show_default=False,
    help="Input data",
)
@click.option(
    "-o",
    "--output_dir",
    required=True,
    type=click.Path(
        exists=True,
        path_type=Path,
        resolve_path=True,
        file_okay=False,
        writable=True,
    ),
    default=Path.cwd(),
    show_default=False,
    callback=create_results,
    help="Output directory",
)
@click.option(
    "--noise_free/--noisy",
    default=True,
    required=True,
)
def convolution(ctx, **kwargs):
    """Multilevel, multidimensional convolution"""
    _convolution(**kwargs)


if __name__ == "__main__":
    cli()
