from __future__ import annotations
from typing import TYPE_CHECKING

from attrs import define, field

import torch
import torch.nn.functional as F
import pennylane as qml

from thesis.quantum.operation.ansatz import Ansatz
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

    def __call__(self, ansatz: type[Ansatz], *args, **kwargs):
        self.ansatz = ansatz.from_dims(*args, **kwargs)
        return super().__call__(self.ansatz)

    def draw(self, include_axis: bool = False):
        plot = super().draw(include_axis=True)
        circuit = qml.draw_mpl(self.ansatz.qnode)()

        fig, ax = zip(plot, circuit)
        return fig, ax if include_axis else fig
