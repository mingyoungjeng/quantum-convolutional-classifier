from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from attrs import define, field
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from qcnn.ml import USE_CUDA
from qcnn.quantum import flatten_array, normalize

if TYPE_CHECKING:
    from typing import Optional, Callable


@define
class Data:
    dataset: type[Dataset] = Dataset
    transform: Optional[Callable] = field(factory=transforms.ToTensor)
    target_transform: Optional[Callable] = None
    classes: Optional[Iterable] = None
    batch_size: tuple[int, int] | int = 0

    def _load(self, is_train: bool = True) -> DataLoader:
        data = self.dataset(
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
        if isinstance(self.batch_size, Iterable):
            batch_size = self.batch_size[0] if is_train else self.batch_size[-1]
        else:
            batch_size = self.batch_size

        dataloader = DataLoader(
            data,
            batch_size=batch_size if batch_size > 0 else len(data),
            shuffle=is_train,
            pin_memory=USE_CUDA,
        )

        return dataloader

    def load(
        self, is_train: Optional[bool] = None
    ) -> DataLoader | tuple[DataLoader, DataLoader]:
        if is_train is None:
            return self._load(True), self._load(False)
        else:
            return self._load(is_train)


@define
class BinaryData(Data):
    classes: Iterable = field(factory=lambda: [0, 1])

    @classes.validator
    def _is_binary(self, _, value):
        if len(value) != 2:
            raise ValueError("Binary data must have only two classes")


def image_transform(dims: Iterable[int], fix_bands=True, flatten=True, norm=True):
    ops = [transforms.Resize(dims[:2]), transforms.ToTensor()]

    if fix_bands and len(dims) >= 3:
        ops += [transforms.Lambda(lambda x: np.moveaxis(np.squeeze(x).numpy(), 0, -1))]

    if flatten:
        ops += [transforms.Lambda(lambda x: flatten_array(x.numpy(), pad=True))]

    if norm:
        ops += [transforms.Lambda(lambda x: normalize(x))]

    return transforms.Compose(ops)
