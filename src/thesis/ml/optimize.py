from __future__ import annotations
from typing import TYPE_CHECKING

from numbers import Number
import torch
from torch import optim
from thesis.ml import create_tensor

if TYPE_CHECKING:
    from typing import Callable, Iterable
    from torch import Tensor
    from torch.utils.data import DataLoader

    LossFunction = CostFunction = Callable[[Iterable[Number], Iterable[Number]], Number]
    MLFunction = Callable[[Iterable[Number], Iterable[Number]], Iterable[Number]]


class Optimizer(optim.Optimizer):
    """
    Extension of torch.optim.Optimizer
    """

    def __new__(self, cls: type[optim.Optimizer], *_, **__):
        """
        Workaround to take a type[optim.Optimizer] as a parameter

        if works and is_readable:
            pass
        else:
            add a comment saying it came to you in a revelation from god and leave it at that
        - Ethan Grantz
        """

        class Opti(self, cls):
            """Don't try this at home, kids"""

        return super().__new__(Opti)

    def __init__(self, cls: type[optim.Optimizer], *args, params=None, **kwargs):
        if params is None:
            cls.__init__(self, [torch.empty(0)], *args, **kwargs)
            self.param_groups.clear()
            self.state.clear()
        else:
            cls.__init__(self, params, *args, **kwargs)

    def __call__(self, params: Tensor | int) -> Optimizer:
        if isinstance(params, Number):
            params = create_tensor(torch.randn, params, requires_grad=True)
        self.add_param_group({"params": params})
        return self

    @property
    def parameters(self) -> Tensor:
        params = [group["params"] and group["params"][0] for group in self.param_groups]
        return params and params[0]


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
    epoch: int = 1,
):
    for i in range(epoch):
        for j, (data, labels) in enumerate(training_dataloader):
            predictions = fn(optimizer.parameters, data)
            backpropagate(predictions, labels, optimizer, cost_fn)

    return optimizer.parameters


@torch.no_grad()
def test(
    fn: MLFunction,
    params: Iterable[Number],
    testing_dataloader: DataLoader,
):
    correct = total = 0
    for data, labels in testing_dataloader:
        predictions = fn(params, data)
        predictions = torch.argmax(predictions, 1)
        correct += torch.count_nonzero(predictions == labels).numpy()
        total += len(data)

    return correct / total
