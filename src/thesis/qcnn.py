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
