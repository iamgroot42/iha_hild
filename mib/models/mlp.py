"""
    Basic N-layer MLPs
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import List


class MLP(nn.Module):
    def __init__(self, layer_depths: List[int], num_classes: int):
        super().__init__()
        self.layer_depths = layer_depths

        num_features = 600
        self.layers = [nn.Linear(num_features, layer_depths[0]), nn.ReLU()]
        for i in range(1, len(layer_depths)):
            self.layers.append(nn.Linear(layer_depths[i-1], layer_depths[i]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(layer_depths[-1], num_classes))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, get_all: bool = False, layer_readout: int = None):
        all_embeds = []
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if layer_readout == i:
                return out
            if get_all and i != len(self.layers) - 1 and i % 2 == 1:
                # Last layer is logits, will collect that anyway
                all_embeds.append(out)

        if get_all:
            return all_embeds

        return out
