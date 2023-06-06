from __future__ import annotations
from typing import TYPE_CHECKING

from attrs import define, field

import pennylane as qml

from thesis.quantum.operation.ansatz import Ansatz
from thesis.ml.model import Model

if TYPE_CHECKING:
    pass


@define
class QCNN(Model):
    ansatz: Ansatz = field(init=False)

    def __call__(self, ansatz: type[Ansatz], *args, silent=False, **kwargs):
        self.ansatz = ansatz.from_dims(*args, **kwargs)
        return super().__call__(self.ansatz, silent=silent)

    def draw(self, include_axis: bool = False, decompose: bool = False):
        plot = super().draw(include_axis=True)
        circuit = self.ansatz.draw(include_axis=include_axis, decompose=decompose)

        fig, ax = zip(plot, circuit)
        return fig, ax if include_axis else fig
