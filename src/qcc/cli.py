from __future__ import annotations
from typing import TYPE_CHECKING

import re
from pathlib import Path
from ast import literal_eval

import click

from qcc.ml.data import Data
from qcc.ml.optimize import Optimizer
from qcc.experiment import Experiment
from qcc.qcnn import QCNN
from qcc.cnn import CNN
from qcc.file import save_dataframe_as_csv, lookup

if TYPE_CHECKING:
    from typing import Any, Iterable, Optional, Callable
    from torch.utils.data import Dataset
    from torch.optim import Optimizer as TorchOptimizer
    from qcc.quantum.operation.ansatz import Ansatz
    from qcc.ml.data import ImageTransform


def run(
    name: str,
    dimensions: Iterable[int],
    ansatz: type[Ansatz],
    dataset: type[Dataset],
    optimizer: type[TorchOptimizer],
    loss: Callable,
    num_trials: int = 1,
    num_layers: int = 0,
    classes: Iterable[int] = (0, 1),
    epoch: int = 1,
    batch_size: int | tuple[int, int] = 1,
    transform: Optional[Callable | type[ImageTransform]] = None,
    ansatz_options: Optional[Iterable[tuple[str, Any]]] = None,
    optimizer_options: Optional[Iterable[tuple[str, Any]]] = None,
    quantum: bool = True,
    verbose: bool = False,
):
    path = Path(f"results/{name}")
    is_quantum = quantum  # clearer notation
    ansatz_options = {} if ansatz_options is None else dict(ansatz_options)
    optimizer_options = {} if optimizer_options is None else dict(optimizer_options)

    # Create model
    cls = QCNN if is_quantum else CNN
    if isinstance(transform, type):
        transform = transform(dimensions, flatten=is_quantum)
    data = Data(dataset, transform, batch_size=batch_size, classes=classes)
    optimizer = Optimizer(optimizer, **optimizer_options)
    loss = loss()
    model = cls.with_logging(data, optimizer, loss, epoch=epoch)

    # Log important values
    model.logger.info(f"Circuit ID: {name}")
    model.logger.info(f"{data=}")
    model.logger.info(f"{optimizer=}")
    model.logger.info(f"{loss=}")

    model.logger.info(f"{num_trials=}")
    model.logger.info(f"{dimensions=}")
    model.logger.info(f"{num_layers=}")
    model.logger.info(f"{epoch=}")
    model.logger.info(f"{ansatz_options=}")

    # Save circuit drawing
    if is_quantum:
        model.ansatz = ansatz.from_dims(
            dimensions, num_layers=num_layers, **ansatz_options
        )
        filename = path.with_stem(f"{name}_circuit").with_suffix(".png")
        model.ansatz.draw(filename=filename, decompose=True)

    # Run experiment
    results_schema=["accuracy", "training_time", "testing_time"]
    experiment = Experiment(model, num_trials, results_schema=results_schema)

    args = (ansatz,) if is_quantum else ()
    results = experiment(
        *args, 
        dimensions, 
        num_layers=num_layers, 
        silent=not verbose, 
        **ansatz_options,
    )

    # Save and print accuracy results
    save_dataframe_as_csv(path.with_suffix(".csv"), results, overwrite=False)
    acc = results["accuracy"]
    msg = f"Accuracy: median={acc.median()}, mean={acc.mean()}, max={acc.max()}, min={acc.min()}, std={acc.std()}"
    model.logger.info(msg)

    # Save aggregated loss history figure
    experiment.draw(path.with_suffix(".png"))


class ObjectType(click.ParamType):
    name = "object"
    root = __name__

    def convert(self, value, param, ctx):
        try:
            return lookup(value, self.root)
        except ValueError:
            self.fail(f"{value!r} is not a valid {self.name}", param, ctx)


class AnsatzType(ObjectType):
    name = "ansatz"
    root = "qcc.quantum.operation.ansatz"


class DatasetType(ObjectType):
    name = "dataset"
    root = "torchvision.datasets"


class OptimizerType(ObjectType):
    name = "optimizer"
    root = "torch.optim"


class LossType(ObjectType):
    name = "loss"
    root = "torch.nn"


class TransformType(ObjectType):
    name = "transform"
    root = "qcc.ml.data"


class TupleType(click.ParamType):
    name = "tuple"

    def __init__(self, cls=type[int]) -> None:
        self.type = cls
        super().__init__()

    def convert(self, value, param, ctx):
        if isinstance(value, (tuple, self.type)):
            return value

        try:
            return tuple(self.type(i) for i in value.split(","))
        except ValueError:
            self.fail(f"{value!r} is not a valid {self.name}", param, ctx)


class OptionType(click.ParamType):
    name = "option"

    def convert(self, value, param, ctx):
        try:
            key, value = re.split(r":|=", value)

            try:
                value = literal_eval(value)
            except ValueError:
                try:
                    value = lookup(value)
                except AttributeError as ex:
                    raise ValueError from ex

            return key, value
        except ValueError:
            self.fail(f"{value!r} is not a valid {self.name}", param, ctx)


@click.command(context_settings={"show_default": True})
@click.option("-n", "--name", required=True, type=str)
@click.option("--ansatz", default="ConvolutionPoolingAnsatz", type=AnsatzType())
@click.option("-d", "--dimensions", default=(28, 28), type=TupleType(int))
@click.option("-l", "--num_layers", default=0, type=int)
@click.option("-ao", "--ansatz_options", type=OptionType(), multiple=True)
@click.option("--dataset", default="MNIST", type=DatasetType())
@click.option("-c", "--classes", default=(0, 1), type=TupleType(int))
@click.option("--transform", default="ImageTransform", type=TransformType())
@click.option("--optimizer", default="Adam", type=OptimizerType())
@click.option("-e", "--epoch", default=1, type=int)
@click.option("-b", "--batch_size", default=1, type=TupleType(int))
@click.option("--loss", "--cost", default="CrossEntropyLoss", type=LossType())
@click.option("-oo", "--optimizer_options", type=OptionType(), multiple=True)
@click.option("-t", "--num_trials", default=1, type=int)
@click.option("--quantum/--classical", default=True)
@click.option("--verbose", is_flag=True)
def cli(**kwargs):
    run(**kwargs)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
