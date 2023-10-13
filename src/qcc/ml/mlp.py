from __future__ import annotations
from typing import TYPE_CHECKING

from torch import nn
import numpy as np

if TYPE_CHECKING:
    pass


class MultiLayerPerceptron(nn.Sequential):
    def __init__(self, dims, num_layers=1, num_features=None, num_classes=2):
        if num_features is None:
            num_features = np.prod(dims)

        lst = [nn.Flatten()]
        for i in range(num_layers):
            # ReLU
            if i > 0:
                lst += [nn.ReLU()]

            lst += [
                nn.Linear(
                    in_channels=np.prod(dims) if i == 0 else num_features,
                    out_channels=num_classes if i == num_layers - 1 else num_features,
                )
            ]

        super().__init__(*lst)

    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
