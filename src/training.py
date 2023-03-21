from typing import Callable, Sequence, Union
from numbers import Number

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Optimizer


# TODO: detect a plateau
def train(
    fn: Callable[[Sequence[Number], Sequence[Number]], Sequence[Number]],
    optimizer: Optimizer,
    training_dataloader: DataLoader,
    cost_fn: Callable[[Sequence[Number], Sequence[Number]], Number],
    initial_parameters: Union[Tensor, None] = None,
    total_params: int = 0,
):
    params = (
        torch.randn(
            total_params, pin_memory=torch.cuda.is_available(), requires_grad=True
        )
        if initial_parameters is None
        else initial_parameters
    )
    opt = optimizer([params], lr=0.01, momentum=0.9, nesterov=True)

    print(f"{len(params)=}")
    for i, (data, labels) in enumerate(training_dataloader):
        opt.zero_grad()
        predictions = fn(params, data)
        cost = cost_fn(predictions, labels)
        cost.backward()
        opt.step()

        # if (i == 0) or ((i + 1) % 100 == 0) or (i + 1 == len(training_dataloader)):
        #     print(f"{i+1}/{len(training_dataloader)}: {cost=:.03f}")

    return params


@torch.no_grad()
def test(
    fn: Callable[[Sequence[Number], Sequence[Number]], Sequence[Number]],
    params: Sequence[Number],
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
