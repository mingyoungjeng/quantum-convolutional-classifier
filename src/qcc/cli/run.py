from __future__ import annotations
from typing import TYPE_CHECKING, Any

from pathlib import Path
from attrs import define, field

from qcc.ml.data import Data, ImageTransform
from qcc.ml.optimize import Optimizer
from qcc.experiment import Experiment
from qcc.qcnn import QCNN
from qcc.cnn import CNN
from qcc.file import load_toml, new_dir, lookup

if TYPE_CHECKING:
    from typing import Any, Iterable, Optional, Callable, Mapping
    from torch.utils.data import Dataset
    from torch.optim import Optimizer as TorchOptimizer
    from qcc.quantum.operation.ansatz import Ansatz


if TYPE_CHECKING:
    pass


def str_to_mod(x):
    return None if x is None else lookup(x)


@define(kw_only=True)
class CLIParameters:
    name: str
    ansatz: Optional[type[Ansatz]] = field(converter=str_to_mod, default=None)
    dataset: type[Dataset] = field(converter=str_to_mod)
    optimizer: type[TorchOptimizer] = field(converter=str_to_mod)
    loss: Callable | type[ImageTransform] = field(converter=str_to_mod)
    transform: Optional[Callable] = field(converter=str_to_mod)
    dimensions: Optional[Iterable[int]] = (16, 16)
    num_trials: Optional[int] = 1
    num_layers: Optional[int] = 0
    classes: Optional[Iterable[int]] = (0, 1)
    epoch: int = 1
    batch_size: Optional[int | tuple[int, int]] = 1
    ansatz_options: Mapping = dict()
    optimizer_options: Mapping = dict()
    output_dir: Path = Path.cwd() / "results"
    is_quantum: bool = True
    verbose: bool = False

    @classmethod
    def from_toml(cls, filename: Path, **kwargs) -> CLIParameters:
        kwargs["name"] = filename.stem
        kwargs = load_toml(filename) | kwargs
        return cls(**kwargs)

    def __call__(self) -> None:
        path = new_dir(self.output_dir / self.name, overwrite=True)
        if isinstance(self.transform, type):
            transform = self.transform.is_quantum(self.dimensions, self.is_quantum)
        else:
            transform = self.transform

        # Create model
        cls = QCNN if self.is_quantum else CNN
        data = Data(
            self.dataset,
            transform,
            batch_size=self.batch_size,
            classes=self.classes,
        )
        optimizer = Optimizer(self.optimizer, **self.optimizer_options)
        loss = self.loss()
        model = cls.with_logging(data, optimizer, loss, epoch=self.epoch)

        # Log important values
        model.logger.info(f"Circuit ID: {self.name}")
        model.logger.info(f"{data=}")
        model.logger.info(f"{optimizer=}")
        model.logger.info(f"{loss=}")

        model.logger.info(f"{self.num_trials=}")
        model.logger.info(f"{self.dimensions=}")
        model.logger.info(f"{self.num_layers=}")
        model.logger.info(f"{self.epoch=}")

        # Save circuit drawing
        if self.is_quantum:
            model.logger.info(f"{self.ansatz_options=}")
            a: Ansatz = self.ansatz.from_dims(
                self.dimensions, num_layers=self.num_layers, **self.ansatz_options
            )
            filename = path / f"{self.name}_circuit.png"
            a.draw(filename=filename, decompose=True)

        # Run experiment
        results_schema = ["accuracy", "training_time", "testing_time"]
        experiment = Experiment(model, self.num_trials, results_schema=results_schema)

        fn = experiment.callable_wrapper(
            *(self.ansatz,) if self.is_quantum else (),
            self.dimensions,
            num_layers=self.num_layers,
            silent=not self.verbose,
            **self.ansatz_options if self.is_quantum else {},
        )
        results = experiment(fn=fn, filename=path / self.name)

        # Print accuracy results
        for name in results.columns:
            col = results[name]
            msg = f"{name}: median={col.median()}, mean={col.mean()}, max={col.max()}, min={col.min()}, std={col.std()}"
            model.logger.info(msg)

        # Save aggregated loss history figure
        experiment.draw(path / f"{self.name}.png")
