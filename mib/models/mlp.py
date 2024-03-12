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

    def forward(
        self,
        x,
        get_all: bool = False,
    ):
        all_embeds = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if get_all and i != len(self.layers) - 1:
                # Last layer is logits, will collect that anyway
                all_embeds.append(x)

        if get_all:
            return all_embeds

        return x
