from typing import Sequence, Callable
from numbers import Number

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.optim import Optimizer
import pennylane as qml
from kuarq import to_qubits, flatten_array

# from pennylane import numpy as np
from pennylane.templates.embeddings import AmplitudeEmbedding
from data import load_dataset
from training import train, test
from ansatz.ansatz import Ansatz

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class QCNN:
    def __init__(
        self,
        dims: Sequence[int],
        ansatz: Ansatz = None,
        classes: Sequence[int] = None,
    ) -> None:
        self.rng = np.random.default_rng()
        self.dims = dims
        self.dims_q = to_qubits(dims)
        self.num_qubits = sum(self.dims_q)

        self.num_layers = 1  # min(self.dims_q)
        self.classes = [0, 1] if classes is None else classes
        if self.num_qubits < len(self.classes):
            print(f"Error: Not enough qubits to represent all classes")

        self.ansatz = Ansatz(self.dims_q) if ansatz is None else ansatz
        device = qml.device("default.qubit", wires=self.num_qubits)
        self.qnode = qml.QNode(self.circuit, device, interface="torch")

        print(f"Using {DEVICE} device")

    def circuit(
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

        # return result

        # Parity implementation
        num_classes = len(self.classes)
        predictions = torch.empty((len(result), num_classes), pin_memory=USE_CUDA)

        for i, probs in enumerate(result):
            num_rows = torch.tensor([len(probs) // num_classes] * num_classes)
            num_rows[: len(probs) % num_classes] += 1

            pred = F.pad(probs, (0, max(num_rows) * num_classes - len(probs)))
            pred = probs.reshape(max(num_rows), num_classes)
            pred = torch.sum(pred, 0)
            pred /= num_rows
            pred /= sum(pred)

            predictions[i] = pred

        return predictions

    def __call__(
        self,
        dataset: Dataset,
        optimizer: Optimizer,
        cost_fn: Callable[[Sequence[Number], Sequence[Number]], Number],
    ):
        transform = transforms.Compose(
            [
                transforms.Resize(self.dims),
                transforms.Lambda(lambda x: flatten_array(np.squeeze(x), pad=True)),
                transforms.ToTensor(),
            ]
        )
        training_dataloader, testing_dataloader = load_dataset(
            dataset, transform, batch_size=4, classes=self.classes
        )

        parameters = torch.empty(0, pin_memory=USE_CUDA, requires_grad=True)
        for num_layers in np.arange(self.ansatz.max_layers + 1):
            self.num_layers = num_layers

            new_params = torch.randn(
                self.ansatz.total_params(self.num_layers),
                pin_memory=USE_CUDA,
                requires_grad=True,
            )

            with torch.no_grad():
                new_params *= 2 * torch.pi

                # if len(parameters) > 0:
                #     new_params[-len(parameters) :] = parameters
                # new_params[: len(parameters)] = parameters

            parameters = train(
                self.predict,
                optimizer,
                training_dataloader,
                cost_fn,
                initial_parameters=new_params,
            )

            accuracy = test(self.predict, parameters, testing_dataloader)
            print(f"{num_layers=}, {accuracy=:.3%}")

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
