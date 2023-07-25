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
    from qcc.ml import MLFunction, CostFunction


@define
class Model:
    data: Data
    optimizer: Optimizer
    cost_fn: CostFunction
    epoch: int = 1
    logger: Optional[Logger] = None

    def _cost(self, *args, **kwargs):
        cost = self.cost_fn(*args, **kwargs)
        self.logger.log(cost, silent=True)
        return cost

    def __call__(self, model: Module, params=None, silent=False):
        # Load dataset
        training_dataloader, testing_dataloader = self.data.load()

        opt = self.optimizer(model.parameters() if params is None else params)
        self.logger.info(f"Number of Parameters: {opt.num_parameters}", silent=silent)

        training_time = time.perf_counter()
        parameters = train(model, opt, training_dataloader, self._cost, self.epoch)
        training_time = time.perf_counter() - training_time
        self.logger.info(f"Training took {training_time:.05} sec", silent=silent)

        testing_time = time.perf_counter()
        accuracy = test(model, testing_dataloader, parameters)
        testing_time = time.perf_counter() - testing_time

        self.logger.info(f"Testing took: {testing_time:.05} sec", silent=silent)
        self.logger.info(f"Accuracy: {accuracy:.05%}", silent=silent)

        return accuracy, training_time, testing_time

    @classmethod
    def with_logging(cls, *args, name: Optional[str] = None, **kwargs):
        if name is None:
            name = cls.__name__.lower()

        schema = [("cost", float)]
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
