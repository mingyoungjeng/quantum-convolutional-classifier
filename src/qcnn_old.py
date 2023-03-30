from qcnn import QCNN
from ansatz.related_work import OldAnsatz

from typing import Sequence, Callable
from numbers import Number

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.optim import Optimizer
from data import load_dataset
from training import train, test


class QCNN_Old(QCNN):
    def __init__(
        self,
        dims: Sequence[int],
        classes: Sequence[int] = None,
    ) -> None:
        super().__init__(dims, ansatz=OldAnsatz(), classes=classes)

    def __call__(
        self,
        dataset: Dataset,
        optimizer: Optimizer,
        cost_fn: Callable[[Sequence[Number], Sequence[Number]], Number],
    ):
        transform = transforms.Compose(
            [
                transforms.Resize(self.dims),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.flatten(torch.squeeze(x).T)),
            ]
        )
        training_dataloader, testing_dataloader = load_dataset(
            dataset, transform, batch_size=4, classes=self.classes
        )

        new_params = torch.randn(
            self.ansatz.total_params(self.dims_q), requires_grad=True
        )
        parameters = train(
            self.predict,
            optimizer,
            training_dataloader,
            cost_fn,
            initial_parameters=new_params,
        )

        accuracy = test(self.predict, parameters, testing_dataloader)

        return accuracy


# if __name__ == "__main__":
#     from torchvision.datasets import MNIST
#     from torch.optim import SGD
#     from torch.nn import CrossEntropyLoss
#     from pennylane import NesterovMomentumOptimizer

#     qcnn = QCNN((16, 16))
#     qcnn.num_layers = 2
#     accuracy = qcnn.run(MNIST, SGD, CrossEntropyLoss())

#     print(accuracy)
