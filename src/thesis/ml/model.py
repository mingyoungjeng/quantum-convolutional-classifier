from __future__ import annotations
from typing import TYPE_CHECKING

from attrs import define
from thesis.ml.data import Data
from thesis.ml.optimize import Optimizer, train, test
from thesis.logger import Logger

if TYPE_CHECKING:
    from typing import Optional
    from pathlib import Path
    from thesis.ml import MLFunction, CostFunction


@define
class Model:
    data: Data
    optimizer: Optimizer
    cost_fn: CostFunction
    logger: Optional[Logger] = None

    def __call__(self, model: MLFunction, params):
        # Load dataset
        training_dataloader, testing_dataloader = self.data.load()

        opt = self.optimizer(params)
        parameters = train(model, opt, training_dataloader, self.cost_fn)

        accuracy = test(model, parameters, testing_dataloader)
        self.logger.logger.info(f"Accuracy: {accuracy:.03%}")

        return accuracy

    @classmethod
    def with_logging(cls, name: str, *args, **kwargs):
        schema = [("loss", float)]
        logger = Logger.from_schema(schema, name=name)
        return cls(*args, **kwargs, logger=logger)

    def save(self, filename: Optional[Path] = None):
        # TODO: Save loss history

        self.logger.save(filename)
