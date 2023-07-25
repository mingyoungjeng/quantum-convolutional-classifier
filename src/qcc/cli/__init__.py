from __future__ import annotations
from typing import TYPE_CHECKING

import re
from ast import literal_eval
from pathlib import Path
import logging

import click

from qcc.cli.run import CLIParameters

if TYPE_CHECKING:
    from typing import Iterable, Any

log = logging.getLogger(__name__)


class ObjectType(click.ParamType):
    name = "object"


class DictType(click.ParamType):
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


def create_results(ctx, param, value):
    results_dir = value / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return value


@click.group()
@click.pass_context
def cli(ctx: click.Context):
    """TODO: put in options for log levels"""

    # Set context settings
    ctx.show_default = True


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
    "--ansatz_options",
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
def run(**kwargs):
    _run(**kwargs)


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
    "-p",
    "--pattern",
    required=True,
    type=str,
    default="**/*.toml",
    help="Config file RegEx pattern",
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
def from_file(ctx, paths: Iterable[Path], pattern: str, output_dir: Path):
    # TODO: handle quantum, classical in TOML
    suffix = ""

    toml_files = set()
    for path in paths:
        if path.is_dir():
            toml_files.update(path.glob(pattern))
        else:
            if path.suffix == suffix:
                toml_files.add(path)

    cmds = (CLIParameters.from_toml(toml) for toml in toml_files)
    for cmd in cmds:
        try:
            cmd()
        except BaseException as e:  # TODO: this is lazy
            log.error(e)


def _setup_module(root: str, obj: str = None):
    if obj is None or "." in obj:
        return obj

    return f"{root}.{obj}"


def _run(
    ansatz: str,
    dataset: str,
    optimizer: str,
    loss: str,
    transform: str,
    *,
    quantum: bool,
    **kwargs,
):
    kwargs["ansatz"] = _setup_module("qcc.quantum.operation.ansatz", ansatz)
    kwargs["dataset"] = _setup_module("torchvision.datasets", dataset)
    kwargs["optimizer"] = _setup_module("torch.optim", optimizer)
    kwargs["loss"] = _setup_module("torch.nn", loss)
    kwargs["transform"] = _setup_module("qcc.ml.data", transform)
    kwargs["is_quantum"] = quantum

    cmd = CLIParameters(**kwargs)
    cmd()


if __name__ == "__main__":
    cli()
