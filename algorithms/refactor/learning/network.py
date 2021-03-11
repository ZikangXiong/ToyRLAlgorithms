from typing import List

import torch as th
from torch import nn


class MLP(nn.Module):

    def __init__(self, layers: List):
        super(MLP, self).__init__()
        self.layers = layers
        layers_no_activation = [layer for layer in layers if hasattr(layer, "parameters")]
        self.model = th.nn.ModuleList(layers_no_activation)

    def forward(self, x):
        _out = x
        for layer in self.layers:
            _out = layer(_out)

        return _out.view(x.size(0), -1)
