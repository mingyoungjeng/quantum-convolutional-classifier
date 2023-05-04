from typing import Callable, Sequence, Tuple, Any
from numbers import Number

import numpy as np

from fn.quantum import flatten_array

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Optimizer
from torchvision import transforms
import torch.nn.functional as F

LossFunction = CostFunction = Callable[[Sequence[Number], Sequence[Number]], Number]
MLFunction = Callable[[Sequence[Number], Sequence[Number]], Sequence[Number]]

USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


def create_tensor(fn: type[Tensor] | Callable, /, *args, **kwargs):
    tensor = fn(*args, **kwargs)
    return tensor.cuda() if USE_CUDA else tensor


def create_optimizer(
    optimizer: type[Optimizer],
    params: Sequence[Number] | Number,
    *args,
    **kwargs,
) -> Optimizer:
    if isinstance(params, Number):
        params = create_tensor(torch.randn, params, requires_grad=True)
    return optimizer([params], *args, **kwargs)


def image_transform(dims: Sequence[int]):
    return transforms.Compose(
        [
            transforms.Resize(dims),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: flatten_array(np.squeeze(x), pad=True)),
        ]
    )


def _dataset(
    dataset: type[Dataset],
    transform=None,
    classes: Sequence[Any] = None,
    batch_size: int = 1,
    is_train: bool = True,
) -> DataLoader:
    data = dataset(
        root="data",
        train=is_train,
        download=True,
        transform=transform,
    )

    if classes is not None:
        (idx,) = np.isin(data.targets, classes).nonzero()
        data = Subset(data, idx)

    dataloader = DataLoader(
        data,
        batch_size=batch_size if is_train else 40,
        shuffle=is_train,
        pin_memory=USE_CUDA,
    )

    return dataloader


def load_dataset(
    dataset: type[Dataset],
    transform: Any = None,
    classes: Sequence[Any] = None,
    batch_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    training_dataloader = _dataset(dataset, transform, classes, batch_size, True)
    testing_dataloader = _dataset(dataset, transform, classes, batch_size, False)

    return training_dataloader, testing_dataloader


def backpropagate(
    predictions: Sequence[Number],
    labels: Sequence[Number],
    opt: Optimizer,
    cost_fn: Callable,
):
    opt.zero_grad()
    cost = cost_fn(predictions, labels)
    cost.backward()
    opt.step()

    return cost


def train(
    fn: MLFunction,
    opt: Optimizer,
    training_dataloader: DataLoader,
    cost_fn: CostFunction,
):
    params = opt.param_groups[0]["params"][0]
    for i, (data, labels) in enumerate(training_dataloader):
        predictions = fn(params, data)
        backpropagate(predictions, labels, opt=opt, cost_fn=cost_fn)

    return params


@torch.no_grad()
def test(
    fn: MLFunction,
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


def parity(result, num_classes: int = 2):
    predictions = create_tensor(torch.empty, (len(result), num_classes))

    for i, probs in enumerate(result):
        num_rows = create_tensor(
            torch.tensor, [len(probs) // num_classes] * num_classes
        )
        num_rows[: len(probs) % num_classes] += 1

        pred = F.pad(probs, (0, max(num_rows) * num_classes - len(probs)))
        pred = probs.reshape(max(num_rows), num_classes)
        pred = torch.sum(pred, 0)
        pred /= num_rows
        pred /= sum(pred)

        predictions[i] = pred

    return predictions
