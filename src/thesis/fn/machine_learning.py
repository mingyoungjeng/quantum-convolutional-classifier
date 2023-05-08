from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
from itertools import chain, pairwise
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torch.multiprocessing import Pool
from torchvision import transforms
from thesis.fn.quantum import flatten_array

if TYPE_CHECKING:
    from typing import Callable, Iterable, Tuple, Optional, Any
    from numbers import Number
    from torch import Tensor
    from torch.utils.data import Dataset
    from torch.optim import Optimizer

    LossFunction = CostFunction = Callable[[Iterable[Number], Iterable[Number]], Number]
    MLFunction = Callable[[Iterable[Number], Iterable[Number]], Iterable[Number]]

USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


def is_iterable(x: Any):
    return hasattr(x, "__iter__")


def cut(
    arr: Iterable[Number], i: Number | Iterable[Number]
) -> Iterable[Iterable[Number]]:
    """
    Splits sequence at indice(s) i

    Args:
        arr (Iterable[Number]): Iterable to split
        i (Number | Iterable[Number]): indice(s) to divide by

    Returns:
        Tuple[Iterable[Number], Iterable[Number]]: split sub-sequences
    """
    if not is_iterable(i):
        i = (i,)

    index = chain((None,), i, (None,))
    return tuple(arr[a:b] for a, b, in pairwise(index))


def create_tensor(fn: type[Tensor] | Callable, /, *args, **kwargs):
    tensor = fn(*args, **kwargs)
    return tensor.cuda() if USE_CUDA else tensor


def create_optimizer(
    optimizer: type[Optimizer],
    params: Iterable[Number] | Number,
    *args,
    **kwargs,
) -> Optimizer:
    if not is_iterable(params):
        params = create_tensor(torch.randn, params, requires_grad=True)
    return optimizer([params], *args, **kwargs)


def image_transform(dims: Iterable[int]):
    return transforms.Compose(
        [
            transforms.Resize(dims),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: flatten_array(np.squeeze(x), pad=True)),
        ]
    )


@dataclass(slots=True)
class DatasetOptions:
    transform: Optional[Callable] = field(default_factory=transforms.ToTensor)
    target_transform: Optional[Callable] = None
    classes: Optional[Iterable[Any]] = None
    batch_size: int | tuple[int, int] = 1

    def _load(self, dataset: type[Dataset], is_train: bool = True) -> DataLoader:
        data = dataset(
            root="data",
            train=is_train,
            download=True,
            transform=self.transform,
            target_transform=self.target_transform,
        )

        if self.classes is not None:
            (idx,) = np.isin(data.targets, self.classes).nonzero()
            data = Subset(data, idx)

        # Handle different batch sizes between train/test
        if is_iterable(self.batch_size):
            batch_size = batch_size[0] if is_train else batch_size[-1]
        else:
            batch_size = self.batch_size

        dataloader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=is_train,
            pin_memory=USE_CUDA,
        )

        return dataloader

    def load(
        self, dataset: type[Dataset], is_train: Optional[bool] = None
    ) -> DataLoader | tuple[DataLoader, DataLoader]:
        if is_train is None:
            return self._load(dataset, True), self._load(dataset, False)
        else:
            return self._load(dataset, is_train)


def backpropagate(
    predictions: Iterable[Number],
    labels: Iterable[Number],
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
    params = opt.param_groups[0]["params"][0]  # TODO: clean up
    for i, (data, labels) in enumerate(training_dataloader):
        predictions = fn(params, data)
        backpropagate(predictions, labels, opt=opt, cost_fn=cost_fn)

    return params


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
