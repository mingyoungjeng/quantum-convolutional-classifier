from __future__ import annotations
from typing import TYPE_CHECKING

from attrs import define
from thesis.ml.data import Data
from thesis.ml.optimize import Optimizer, train, test
from thesis.experiment.logger import Logger
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from typing import Optional
    from pathlib import Path
    from thesis.ml import MLFunction, CostFunction


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

    def __call__(self, model: MLFunction, params):
        # Load dataset
        training_dataloader, testing_dataloader = self.data.load()

        opt = self.optimizer(params)
        self.logger.logger.info(f"Number of Parameters: {opt.num_parameters}")

        parameters = train(model, opt, training_dataloader, self._cost, self.epoch)

        accuracy = test(model, parameters, testing_dataloader)
        self.logger.logger.info(f"Accuracy: {accuracy:.03%}")

        return accuracy

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

        return fig, ax if include_axis else fig

    def save(self, filename: Optional[Path] = None):
        # TODO: Save loss history

        self.logger.save(filename)
