from __future__ import annotations
from typing import TYPE_CHECKING

import time
from attrs import define
from torch.nn import Module
from qcc.ml.data import Data
from qcc.ml.optimize import Optimizer, train, test
from qcc.experiment.logger import Logger
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing import Optional
    from pathlib import Path
    from qcc.ml import CostFunction


@define
class Model:
    module: Module
    data: Data
    optimizer: Optimizer
    cost_fn: CostFunction
    epoch: int = 1
    logger: Optional[Logger] = None

    def _cost(self, *args, **kwargs):
        cost = self.cost_fn(*args, **kwargs)
        self.logger.log(cost=cost, silent=True)
        return cost

    def __call__(self, params=None, silent=False):
        if hasattr(self.module, "reset_parameters"):
            self.module.reset_parameters()

        # Load dataset
        training_dataloader, testing_dataloader = self.data.load()

        opt = self.optimizer(self.module.parameters() if params is None else params)
        self.logger.info(f"Number of Parameters: {opt.num_parameters}", silent=silent)

        training_time = 0
        testing_time = 0
        for i in range(self.epoch):
            now = time.perf_counter()
            parameters = train(self.module, opt, training_dataloader, self._cost)
            training_time += time.perf_counter() - now
            self.logger.info(
                f"(Epoch {i+1}) Training took {training_time:.05} sec", silent=silent
            )

            now = time.perf_counter()
            accuracy = test(self.module, testing_dataloader, parameters)
            testing_time = time.perf_counter() - now

            self.logger.info(
                f"(Epoch {i+1}) Testing took: {testing_time:.05} sec", silent=silent
            )
            self.logger.info(f"(Epoch {i+1}) Accuracy: {accuracy:.05%}", silent=silent)

            self.logger.log(
                accuracy=accuracy,
                training_time=training_time,
                testing_time=testing_time,
                silent=True,
            )

        return accuracy, training_time, testing_time

    @classmethod
    def with_logging(cls, *args, name: Optional[str] = None, **kwargs):
        if name is None:
            name = type(args[0]).__name__.lower()

        schema = [
            ("cost", float),
            ("accuracy", float),
            ("training_time", float),
            ("testing_time", float),
        ]
        fmt = "%(asctime)s: (%(name)s) %(message)s"
        logger = Logger.from_schema(schema, name, fmt)
        return cls(*args, **kwargs, logger=logger)

    def draw(self, include_axis: bool = False):
        if self.logger is None:
            return

        fig, ax = plt.subplots()
        cost = self.logger.df.get_column("cost").to_numpy()
        ax.plot(cost)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")

        return (fig, ax) if include_axis else fig

    def save(self, filename: Optional[Path] = None):
        # TODO: Save loss history

        self.logger.save(filename)
