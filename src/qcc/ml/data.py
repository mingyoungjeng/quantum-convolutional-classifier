from __future__ import annotations
from typing import TYPE_CHECKING, Iterable

from attrs import define, field
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from qcc.ml import USE_CUDA
from qcc.quantum import flatten_array, normalize

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


class ImageTransform(transforms.Compose):
    def __init__(
        self,
        dims: Optional[Iterable[int]] = None,
        fix_bands=True,
        flatten=True,
        norm=True,
        squeeze=True,
    ):
        ops = [transforms.ToTensor()]

        if dims is not None:
            ops = [transforms.Resize(dims[:2]), *ops]

        if squeeze:
            ops += [transforms.Lambda(torch.squeeze)]

        if fix_bands and len(dims) >= 3:
            ops += [self._fix_bands()]

        if flatten:
            ops += [self._flatten()]

        if norm:
            ops += [self._norm()]

        super().__init__(ops)

    def __repr__(self) -> str:
        return "image_transform"

    @staticmethod
    def _fix_bands():
        return transforms.Lambda(lambda x: torch.moveaxis(x, 0, -1))

    @staticmethod
    def _flatten():
        return transforms.Lambda(lambda x: flatten_array(x.numpy(), pad=True))

    @staticmethod
    def _norm():
        return transforms.Lambda(normalize)
    
    @classmethod
    def is_quantum(cls, dims: Optional[Iterable[int]] = None, is_quantum: bool = True):
        return cls(dims, fix_bands=is_quantum, flatten=is_quantum, norm=is_quantum, squeeze=is_quantum)


class ImageTransform1D(ImageTransform):
    def __repr__(self) -> str:
        return "image_transform_1D"

    @staticmethod
    def _flatten():
        return transforms.Lambda(torch.flatten)
