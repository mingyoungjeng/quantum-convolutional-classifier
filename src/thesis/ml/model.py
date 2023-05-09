from __future__ import annotations
from typing import TYPE_CHECKING

from attrs import define
from thesis.ml.data import Data
from thesis.ml.optimize import Optimizer, train, test

if TYPE_CHECKING:
    from thesis.ml.ml import MLFunction, CostFunction


@define
class Model:
    data: Data
    optimizer: Optimizer
    cost_fn: CostFunction

    def __call__(self, model: MLFunction, params):
        # Load dataset
        training_dataloader, testing_dataloader = self.data.load()

        opt = self.optimizer(params)
        parameters = train(model, opt, training_dataloader, self.cost_fn)

        accuracy = test(model, parameters, testing_dataloader)

        return accuracy
