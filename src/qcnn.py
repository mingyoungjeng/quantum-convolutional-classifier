from typing import Sequence
from numbers import Number

import numpy as np

# from pennylane import numpy as np
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
from ansatz.ansatz import Ansatz

import torch
from torch.utils.data import Dataset
from torch.optim import Optimizer

# RNG = np.random.default_rng()

from fn.quantum import to_qubits
from fn.machine_learning import (
    CostFunction,
    load_dataset,
    train,
    test,
    create_optimizer,
    image_transform,
    parity,
)


class QCNN:
    def __init__(
        self,
        dims: Sequence[int],
        ansatz: type[Ansatz],
        classes: Sequence[int] = None,
    ) -> None:
        self.dims = dims
        self.dims_q = to_qubits(dims)
        self.num_qubits = sum(self.dims_q)
        self.num_layers = 1

        self.classes = [0, 1] if classes is None else classes
        if self.num_qubits < len(self.classes):
            print("Error: Not enough qubits to represent all classes")

        self.ansatz = ansatz(self.num_qubits)
        device = qml.device("default.qubit", wires=self.num_qubits)
        self.qnode = qml.QNode(self._circuit, device, interface="torch")

    def _circuit(
        self, params: Sequence[Number], psi_in: Sequence[Number] = None
    ) -> Sequence[float]:
        # c2q
        if psi_in is None:
            psi_in = np.asarray([1, 0])

        AmplitudeEmbedding(psi_in, range(self.num_qubits), pad_with=0, normalize=True)
        meas = self.ansatz(params, self.num_layers)

        # q2c
        return qml.probs(meas)

    def predict(self, *args, **kwargs) -> torch.Tensor:
        result = self.qnode(*args, **kwargs)

        if result.dim() == 1:  # Makes sure batch is 2D array
            result = result.unsqueeze(0)

        return parity(result)  # result

    # TODO: num_layers as an option
    # TODO: batch_size as an option
    def __call__(
        self,
        dataset: type[Dataset],
        optimizer: type[Optimizer],
        cost_fn: CostFunction,
    ):
        transform = image_transform(self.dims)  # TODO: transform as option
        training_dataloader, testing_dataloader = load_dataset(
            dataset, transform, batch_size=4, classes=self.classes
        )

        # TODO: optimizer options should be an option
        opt = create_optimizer(
            optimizer,
            self.ansatz.total_params(),
            lr=0.01,
            momentum=0.9,
            nesterov=True,
        )
        parameters = train(self.predict, opt, training_dataloader, cost_fn)

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
