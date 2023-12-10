"""
Model
A (hopefully) library-agnostic method of performing supervised learning.
"""

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
    """A (hopefully) library-agnostic method of performing supervised learning."""

    module: Module
    data: Data
    optimizer: Optimizer
    cost_fn: CostFunction
    epoch: int = 1
    logger: Optional[Logger] = None

    def _cost(self, *args, **kwargs):
        """Evaluate cost/loss and log it for history"""

        cost = self.cost_fn(*args, **kwargs)
        self.logger.log(cost=cost, silent=True)
        return cost

    def __call__(self, params=None, silent=False) -> tuple[float, float, float]:
        """
        Perform classification trial

        Args:
            params (optional): Set initial parameters. Defaults to None and takes parameters from module.
            silent (bool, optional): Whether to print log of training/testing/accuracy. Defaults to False.

        Returns:
            tuple[float, float, float]: accuracy, training time, and testing time
        """

        if hasattr(self.module, "reset_parameters"):
            self.module.reset_parameters()

        # Load dataset
        training_dataloader, testing_dataloader = self.data.load()

        # Create optimizer
        opt = self.optimizer(self.module.parameters() if params is None else params)
        self.logger.info(f"Number of Parameters: {opt.num_parameters}", silent=silent)

        training_time = 0
        testing_time = 0
        for i in range(self.epoch):
            # ==== Training ==== #
            now = time.thread_time()
            parameters = train(self.module, opt, training_dataloader, self._cost)
            training_time += time.thread_time() - now

            msg = f"Training took {training_time:.5f} sec"
            msg = f"(Epoch {i+1}) {msg}" if self.epoch > 1 else msg
            self.logger.info(msg, silent=silent)

            # ==== Testing ==== #
            now = time.thread_time()
            accuracy = test(self.module, testing_dataloader, parameters)
            testing_time = time.thread_time() - now

            msg = f"Testing took: {testing_time:.5f} sec"
            msg = f"(Epoch {i+1}) {msg}" if self.epoch > 1 else msg
            self.logger.info(msg, silent=silent)

            # ==== Accuracy ==== #
            msg = f"Accuracy: {accuracy:.3%}"
            msg = f"(Epoch {i+1}) {msg}" if self.epoch > 1 else msg
            self.logger.info(msg, silent=silent)

            self.logger.log(
                accuracy=accuracy,
                training_time=training_time,
                testing_time=testing_time,
                silent=True,
            )

        return accuracy, training_time, testing_time

    @classmethod
    def with_logging(cls, *args, name: Optional[str] = None, **kwargs) -> Model:
        """
        Class constructor that generates new Logger

        Args:
            name (Optional[str]): Name of Logger. Defaults to __name__ of first argument.

        Returns:
            Model
        """

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

    # TODO: convert to plotly
    def draw(self, include_axis: bool = False):
        """Create loss history graph"""

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
