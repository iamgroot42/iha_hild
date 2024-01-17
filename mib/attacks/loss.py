"""
    Implementation for LOSS attack and some variants.
"""
import torch as ch
import numpy as np

from mib.attacks.base import Attack
from mib.attacks.attack_utils import compute_scaled_logit


class LOSS(Attack):
    """
    Standard LOSS attack.
    """

    def __init__(self, model):
        super().__init__("LOSS", model)

    @ch.no_grad()
    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        loss = self.criterion(self.model(x.cuda()).detach(), y.cuda())
        return -loss.cpu().numpy()


class Logit(Attack):
    """
    Logit-scaling applied to model confidence
    """

    def __init__(self, model):
        super().__init__("LOSS_LiRA", model)

    @ch.no_grad()
    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        return compute_scaled_logit(self.model, x.cuda(), y.cuda())
