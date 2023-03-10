from typing import Sequence

# import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.optim import Optimizer
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.embeddings import AmplitudeEmbedding
from ansatz_old import qcnn_ansatz
from data import load_dataset
from training import train, test


class QCNN:
    def __init__(self, dims) -> None:
        self.rng = np.random.default_rng()
        self.dims = dims

        dims_q = [int(np.ceil(np.log2(dim))) for dim in dims]
        self.num_qubits = sum(dims_q)

        device = qml.device("default.qubit", wires=np.sum(dims_q))
        self.qnode = qml.QNode(self.circuit, device, interface="torch")

    def circuit(
        self, params: Sequence[float], psi_in: Sequence[float] = None
    ) -> Sequence[float]:
        if psi_in is None:
            psi_in = np.asarray([1, 0])
        wires = range(self.num_qubits)

        # c2q
        AmplitudeEmbedding(psi_in, wires, pad_with=0, normalize=True)

        # meas = qcnn_ansatz(params, self.dims, self.num_layers)
        meas = qcnn_ansatz(params, wires)

        # q2c
        # TODO: subject to change if meas qubits don't need to be entangled
        return qml.probs(meas)

    def cost(self, params, X, Y):
        predictions = [self.qnode(params, x) for x in X]

        loss = 0
        for l, p in zip(Y, predictions):
            c_entropy = l * (np.log(p[l])) + (1 - l) * np.log(1 - p[1 - l])
            loss = loss + c_entropy
        return -1 * loss

    def run(self, dataset: Dataset, optimizer: Optimizer, cost_fn):
        transform = transforms.Compose(
            [
                transforms.Resize(self.dims),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.flatten(torch.squeeze(x).T)),
            ]
        )
        training_dataloader, testing_dataloader = load_dataset(
            dataset, transform, batch_size=4, classes=[0, 1]
        )

        total_params = (15 + 2) * 3
        # params = self.rng.standard_normal(total_params, requires_grad=True)
        # opt = qml.NesterovMomentumOptimizer(stepsize=0.01)
        # loss_history = []

        # for i, (X_batch, Y_batch) in enumerate(training_dataloader):
        #     # print(self.cost(params, X_batch.numpy(), Y_batch.numpy()))
        #     params, cost_new = opt.step_and_cost(
        #         lambda v: self.cost(v, X_batch.numpy(), Y_batch.numpy()), params
        #     )
        #     loss_history.append(cost_new)
        #     if i % 10 == 0:
        #         print(f"iteration: {i}/{len(training_dataloader)}, cost: {cost_new}")

        # return loss_history, params

        optimal_params = train(
            self.qnode, optimizer, training_dataloader, cost_fn, total_params
        )

        accuracy = test(
            self.qnode, optimal_params, testing_dataloader=testing_dataloader
        )
        return accuracy

        # dims = np.stack(
        #     [
        #         np.pad(range(end - size, end), (0, max(dims_q) - size), constant_values=-1)
        #         for size, end in zip(dims_q, np.cumsum(dims_q))
        #     ],
        #     axis=1,
        # )
