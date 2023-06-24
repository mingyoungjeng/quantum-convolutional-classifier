from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from numbers import Number
from math import pi
import torch
from torch import optim
from qcc.ml import create_tensor, USE_CUDA

if TYPE_CHECKING:
    from typing import Callable, Optional
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
            self.reset()
        else:
            cls.__init__(self, params, *args, **kwargs)

        self.args, self.kwargs = args, kwargs

    def __call__(self, params: Tensor | int) -> Optimizer:
        self.reset()
        if isinstance(params, Number):
            params = init_params(params)
        self.add_param_group({"params": params})
        return self

    def reset(self):
        self.param_groups.clear()
        self.state.clear()

    @property
    def parameters(self) -> Tensor | list[Tensor]:
        params = [p for group in self.param_groups for p in group["params"]]
        return params[0] if len(params) == 1 else params

    @property
    def num_parameters(self) -> int:
        params = self.parameters
        if isinstance(params, torch.Tensor):
            params = [params]

        n = sum([param.flatten().shape[0] for param in params])
        return n

    def __repr__(self) -> str:
        return f"{type(self).__bases__[-1]}, {self.args=}, {self.kwargs=}"


def init_params(size, angle=False):
    params = torch.nn.Parameter(create_tensor(torch.randn, size, requires_grad=True))

    if angle:
        with torch.no_grad():
            params *= 2 * pi
    return params


def backpropagate(
    predictions: torch.Tensor,
    labels: torch.Tensor,
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
    params: Optional[Iterable[Number]] = None,
):
    for i in range(epoch):
        for j, (data, labels) in enumerate(training_dataloader):
            if USE_CUDA:
                data, labels = data.cuda(), labels.cuda()
            predictions = fn(data) if params is None else fn(data, params)
            backpropagate(predictions, labels, optimizer, cost_fn)

    return None if params is None else optimizer.parameters


@torch.no_grad()
def test(
    fn: MLFunction,
    testing_dataloader: DataLoader,
    params: Optional[Iterable[Number]] = None,
):
    correct = total = 0
    for data, labels in testing_dataloader:
        if USE_CUDA:
            data, labels = data.cuda(), labels.cuda()
        predictions = fn(data) if params is None else fn(data, params)
        predictions = torch.argmax(predictions, 1)
        correct += torch.count_nonzero(predictions == labels)
        total += len(data)

    return correct / total
