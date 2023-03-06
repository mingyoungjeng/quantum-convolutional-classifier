from typing import Sequence
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pennylane as qml
import numpy as np
from pennylane.templates.embeddings import AmplitudeEmbedding
from ansatz import qcnn_ansatz
from data import load_dataset
from training import train


class QCNN:
    def __init__(self) -> None:
        pass

    def circuit(
        self, params: Sequence[float], psi_in: Sequence[float] = None
    ) -> Sequence[float]:
        # c2q
        if psi_in is None:
            psi_in = np.asarray([1, 0])
        wires = range(self.num_qubits)
        AmplitudeEmbedding(psi_in, wires, pad_with=0, normalize=True)

        # meas = qcnn_ansatz(params, self.dims, self.num_layers)
        meas = qcnn_ansatz(params, wires)

        # q2c
        # TODO: subject to change if meas qubits don't need to be entangled
        result = qml.probs(meas)

        return np.argmax(result)

    def run(self, dataset: Dataset, cost_fn):
        transform = transforms.Compose(
            [
                transforms.Resize(dims),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.flatten(torch.squeeze(x).T)),
            ]
        )
        training_dataloader, testing_dataloader = load_dataset(
            dataset, transform, batch_size=4
        )

        dims_q = [int(np.ceil(np.log2(dim))) for dim in dims]

        device = qml.device("device", wires=np.sum(dims_q))
        qnode = qml.QNode(self.circuit, device, interface="torch")

        optimal_params = train()

        test(optimal_params)

        # dims = np.stack(
        #     [
        #         np.pad(range(end - size, end), (0, max(dims_q) - size), constant_values=-1)
        #         for size, end in zip(dims_q, np.cumsum(dims_q))
        #     ],
        #     axis=1,
        # )
