from __future__ import annotations
from typing import TYPE_CHECKING

from attrs import define, field

import torch
import torch.nn.functional as F
import pennylane as qml

from thesis.operation.ansatz import Ansatz
from thesis.ml.model import Model
from thesis.ml.ml import create_tensor

if TYPE_CHECKING:
    from torch import Tensor


def parity(result, num_classes: int = 2):
    predictions = create_tensor(torch.empty, (len(result), num_classes))

    for i, probs in enumerate(result):
        num_rows = create_tensor(
            torch.tensor, [len(probs) // num_classes] * num_classes
        )
        num_rows[: len(probs) % num_classes] += 1

        pred = F.pad(probs, (0, max(num_rows) * num_classes - len(probs)))
        pred = probs.reshape(max(num_rows), num_classes)
        pred = torch.sum(pred, 0)
        pred /= num_rows
        pred /= sum(pred)

        predictions[i] = pred

    return predictions


@define
class QCNN(Model):
    ansatz: Ansatz = field(init=False)

    def predict(self, *args, **kwargs) -> Tensor:
        result = self.ansatz.qnode(*args, **kwargs)

        if result.dim() == 1:  # Makes sure batch is 2D array
            result = result.unsqueeze(0)

        return parity(result)  # result

    def __call__(self, ansatz: type[Ansatz], *args, **kwargs):
        self.ansatz = ansatz.from_dims(*args, **kwargs)

        return super().__call__(self.predict, self.ansatz.shape)

    @property
    def draw(self):
        return qml.draw_mpl(self.ansatz.qnode)(self.optimizer.parameters)
