from typing import Sequence, Callable
from numbers import Number

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.optim import Optimizer
import pennylane as qml

# from pennylane import numpy as np
from pennylane.templates.embeddings import AmplitudeEmbedding
from ansatz.ansatz import qcnn_ansatz
from data import load_dataset
from training import train, test


class QCNN:
    def __init__(
        self,
        dims: Sequence[int],
        ansatz: Callable = None,
        classes: Sequence[int] = None,
    ) -> None:
        self.rng = np.random.default_rng()
        self.dims = dims
        self.dims_q = [int(np.ceil(np.log2(dim))) for dim in dims]
        self.num_qubits = sum(self.dims_q)

        self.num_layers = 1  # min(self.dims_q)
        self.classes = [0, 1] if classes is None else classes
        if self.num_qubits < len(self.classes):
            print(f"Error: Not enough qubits to represent all classes")

        # TODO: any better formula?
        self.max_layers = min(self.dims_q) + int(
            np.log2(len(self.classes)) // -len(self.dims_q)
        )

        self.ansatz = qcnn_ansatz if ansatz is None else ansatz
        device = qml.device("default.qubit", wires=self.num_qubits)
        self.qnode = qml.QNode(self.circuit, device, interface="torch")

    def circuit(
        self, params: Sequence[Number], psi_in: Sequence[Number] = None
    ) -> Sequence[float]:
        # c2q
        if psi_in is None:
            psi_in = np.asarray([1, 0])
        AmplitudeEmbedding(psi_in, range(self.num_qubits), pad_with=0, normalize=True)

        meas = self.ansatz(
            params,
            self.dims_q,
            num_layers=self.num_layers,
            num_classes=len(self.classes),
        )

        # q2c
        # TODO: subject to change if meas qubits don't need to be entangled
        return qml.probs(meas)

    def run(
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

        parameters = torch.empty(0, requires_grad=True)
        for num_layers in 1 + np.arange(self.max_layers):
            self.num_layers = num_layers

            conv_params = 6 * min(self.dims_q) * self.num_layers
            # pool_params = 3 * len(self.dims_q) * (self.num_layers - 1)
            pool_params = int(
                6
                * len(self.dims_q)
                * (self.num_layers - 1)
                * (
                    self.num_layers / 2
                    - (np.log2(len(self.classes)) // -len(self.dims_q))
                    - 1
                )
            )
            total_params = conv_params + pool_params
            # print(total_params)

            new_params = torch.randn(total_params, requires_grad=True)

            with torch.no_grad():
                new_params *= 2 * torch.pi

                if len(parameters) > 0:
                    new_params[-len(parameters) :] = parameters

            parameters = train(
                self.qnode,
                optimizer,
                training_dataloader,
                cost_fn,
                initial_parameters=new_params,
            )

        accuracy = test(self.qnode, parameters, testing_dataloader)

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
