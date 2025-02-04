from __future__ import annotations
from typing import TYPE_CHECKING

from pathlib import Path
from attrs import define, field

from qcc.ml.model import Model
from qcc.ml.data import Data, ImageTransform
from qcc.ml.optimize import Optimizer
from qcc.experiment import Experiment
from qcc.file import load_toml, new_dir, lookup

if TYPE_CHECKING:
    from typing import Iterable, Callable, Mapping
    from torch.nn import Module
    from torch.utils.data import Dataset
    from torch.optim import Optimizer as TorchOptimizer


def str_to_mod(x):
    return lookup(x) if isinstance(x, str) else x


def dict_converter(x: Mapping):
    for key, value in x.items():
        x[key] = str_to_mod(value)

    return x


@define(kw_only=True)
class Classify:
    name: str
    module: type[Module] = field(converter=str_to_mod)
    dataset: type[Dataset] = field(converter=str_to_mod)
    optimizer: type[TorchOptimizer] = field(converter=str_to_mod)
    loss: Callable | type[ImageTransform] = field(converter=str_to_mod)
    transform: Callable | None = field(converter=str_to_mod)
    dimensions: Iterable[int] | None = (16, 16)
    num_trials: int | None = 1
    num_layers: int | None = 0
    classes: Iterable[int] | None = (0, 1)
    epoch: int = 1
    batch_size: int | tuple[int, int] | None = 1
    module_options: Mapping = field(factory=dict, converter=dict_converter)
    optimizer_options: Mapping = field(factory=dict, converter=dict_converter)
    output_dir: Path | None = None
    is_quantum: bool = True
    verbose: bool = False

    @classmethod
    def from_toml(cls, filename: Path, **kwargs) -> Classify:
        """
        Load from toml config
        """
        kwargs["name"] = filename.stem
        kwargs = load_toml(filename) | kwargs

        if "quantum" in kwargs:
            kwargs["is_quantum"] = kwargs.pop("quantum")
        if "classical" in kwargs:
            kwargs["is_quantum"] = not kwargs.pop("classical")

        return cls(**kwargs)

    def __call__(self) -> None:
        """
        Perform classification experiment from parameter
        """
        if self.output_dir is None:
            filename = None
        else:
            filename = new_dir(self.output_dir / self.name, overwrite=True)
            filename = filename / self.name

        if self.is_quantum:
            self.module = self.module.from_dims

        # Prepare module, data, optimize, loss_fn from parameters
        module: Module = self.module(
            self.dimensions,
            num_layers=self.num_layers,
            **self.module_options,
        )
        data = Data(
            self.dataset,
            self.transform(self.dimensions)
            if isinstance(self.transform, type)
            else self.transform,
            batch_size=self.batch_size,
            classes=self.classes,
        )
        optimizer = Optimizer(self.optimizer, **self.optimizer_options)
        loss: Callable = self.loss()

        # Create a ML model from paramters
        model = Model.with_logging(module, data, optimizer, loss, epoch=self.epoch)

        # Log important values
        model.logger.info(f"Circuit ID: {self.name}")
        model.logger.info(f"{module=}")
        model.logger.info(f"{data=}")
        model.logger.info(f"{optimizer=}")
        model.logger.info(f"{loss=}")

        model.logger.info(f"{self.num_trials=}")
        model.logger.info(f"{self.dimensions=}")
        model.logger.info(f"{self.num_layers=}")
        model.logger.info(f"{self.epoch=}")
        model.logger.info(f"{self.module_options=}")

        # Save circuit drawing (if QML)
        if self.is_quantum and filename is not None:
            draw_path = filename.with_stem(f"{self.name}_circuit")
            module.draw(filename=draw_path, decompose=True)

        # Run experiment
        results_schema = ["accuracy", "training_time", "testing_time"]
        experiment = Experiment(model, self.num_trials, results_schema)
        # experiment.partial(silent=not self.verbose)
        results = experiment(filename=filename)

        # Print accuracy results
        metrics = ("median", "mean", "max", "min", "std")
        for name in results.columns:
            col = results[name]
            msg = (f"{metric}={getattr(col, metric)()}" for metric in metrics)
            msg = ", ".join(msg)
            msg = f"{name}: {msg}"
            model.logger.info(msg)

        # Save aggregated loss history figure
        experiment.draw(filename=filename)

        return self.name, experiment.dfs
