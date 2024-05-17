"""
    Basic N-layer MLPs
"""
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """
    Generic MLP class wrapper.
    """
    def __init__(self, num_features:int, layer_depths: List[int], num_classes: int, add_sigmoid: bool = False):
        super().__init__()
        self.layer_depths = layer_depths
        if add_sigmoid and num_classes != 1:
            raise ValueError(f"Only working with single-class classification. Are you sure you want to add a sigmoid for {num_classes} classes?")

        # Add first layer
        if len(layer_depths) > 0:
            self.layers = [nn.Linear(num_features, layer_depths[0]), nn.ReLU()]
        else:
            self.layers = [nn.Linear(num_features, num_classes)]

        # Add intermediate layers
        for i in range(1, len(layer_depths)):
            self.layers.append(nn.Linear(layer_depths[i-1], layer_depths[i]))
            self.layers.append(nn.ReLU())

        # Final layer
        if len(layer_depths) > 0:
            self.layers.append(nn.Linear(layer_depths[-1], num_classes))
        if add_sigmoid:
            self.layers.append(nn.Sigmoid())
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


class MLPQuadLoss(MLP):
    def __init__(self, num_features:int, layer_depths: List[int], num_classes: int):
        super().__init__(num_features, layer_depths, num_classes, add_sigmoid=True)
