from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field

import numpy as np
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding
import torch
from thesis.fn.quantum import to_qubits
from thesis.fn.machine_learning import (
    DatasetOptions,
    train,
    test,
    create_optimizer,
    image_transform,
    parity,
)

if TYPE_CHECKING:
    from typing import Iterable, Optional, Callable
    from numbers import Number
    from torch.utils.data import Dataset
    from torch.optim import Optimizer
    from pennylane import QNode
    from thesis.operation import Parameters
    from thesis.operation.ansatz import Ansatz
    from thesis.fn.machine_learning import CostFunction


@dataclass(slots=True)
class QCNN:
    dims: Iterable[int] = field(default_factory=list)
    dataset_options: DatasetOptions = DatasetOptions()
    optimizer_options: dict = field(default_factory=dict)

    qnode: Optional[QNode] = field(init=False, repr=False, default=None)
    ansatz: Optional[Ansatz] = field(init=False, repr=False, default=None)

    def _circuit(self, params: Parameters, psi_in: Iterable[Number]) -> Iterable[float]:
        # c2q
        AmplitudeEmbedding(psi_in, self.ansatz.wires, pad_with=0, normalize=True)

        meas = self.ansatz(params)

        # q2c
        return qml.probs(meas)

    def predict(self, *args, **kwargs) -> torch.Tensor:
        result = self.qnode(*args, **kwargs)

        if result.dim() == 1:  # Makes sure batch is 2D array
            result = result.unsqueeze(0)

        return parity(result)  # result

    def __call__(
        self,
        ansatz: type[Ansatz],
        dataset: type[Dataset],
        optimizer: type[Optimizer],
        cost_fn: CostFunction,
    ):
        # Create ansatz and qnode
        self.ansatz = ansatz.from_dims(self.dims)
        device = qml.device("default.qubit", wires=self.ansatz.num_wires)
        self.qnode = qml.QNode(self._circuit, device, interface="torch")

        # Load dataset
        training_dataloader, testing_dataloader = self.dataset_options.load(dataset)

        optimizer = create_optimizer(
            optimizer, self.ansatz.shape, **self.optimizer_options
        )
        parameters = train(self.predict, optimizer, training_dataloader, cost_fn)

        accuracy = test(self.predict, parameters, testing_dataloader)

        return accuracy
