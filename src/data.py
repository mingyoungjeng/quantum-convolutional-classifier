from typing import Sequence, Tuple
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np


def load_dataset(
    dataset: Dataset, transform=None, classes: Sequence = None, batch_size: int = 1
) -> Tuple[DataLoader, DataLoader]:
    for is_train in [True, False]:
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
            data, batch_size=batch_size if is_train else 1, shuffle=is_train
        )

        if is_train:
            training_dataloader = dataloader
        else:
            testing_dataloader = dataloader

    return training_dataloader, testing_dataloader
