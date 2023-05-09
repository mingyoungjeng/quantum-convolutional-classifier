from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from torch import optim
from thesis.ml.ml import create_tensor

if TYPE_CHECKING:
    from typing import Callable, Iterable
    from numbers import Number
    from torch import Tensor
    from torch.utils.data import DataLoader

    LossFunction = CostFunction = Callable[[Iterable[Number], Iterable[Number]], Number]
    MLFunction = Callable[[Iterable[Number], Iterable[Number]], Iterable[Number]]


class Optimizer(optim.Optimizer):
    def __init__(self, cls, *args, **kwargs):
        # TODO: messy, find alternative
        self.__class__ = type(
            self.__class__.__name__, (cls, object), dict(self.__class__.__dict__)
        )

        cls.__init__(self, [torch.empty(0)], *args, **kwargs)
        self.param_groups.clear()
        self.state.clear()

    def __call__(self, params: Tensor | int) -> Optimizer:
        if isinstance(params, int):
            params = create_tensor(torch.randn, params, requires_grad=True)
        self.add_param_group({"params": params})
        return self

    @property
    def parameters(self):
        pass


def backpropagate(
    predictions: Iterable[Number],
    labels: Iterable[Number],
    optimizer: optim.Optimizer,
    cost_fn: Callable,
):
    optimizer.zero_grad()
    cost = cost_fn(predictions, labels)
    cost.backward()
    optimizer.step()

    return cost


def train(
    fn: MLFunction,
    optimizer: Optimizer,
    training_dataloader: DataLoader,
    cost_fn: CostFunction,
):
    for i, (data, labels) in enumerate(training_dataloader):
        predictions = fn(optimizer.parameters, data)
        backpropagate(predictions, labels, optimizer, cost_fn)

    return optimizer.parameters


@torch.no_grad()
def test(
    fn: MLFunction,
    params: Iterable[Number],
    testing_dataloader: DataLoader,
):
    correct = 0
    total = 0

    for data, labels in testing_dataloader:
        predictions = fn(params, data)
        predictions = torch.argmax(predictions, 1)
        correct += torch.count_nonzero(predictions == labels).numpy()
        total += len(data)

    return correct / total
