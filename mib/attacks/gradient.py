"""
    Attack based on gradient norm.
"""
import numpy as np
from mib.attacks.base import Attack
from mib.attacks.attack_utils import compute_gradients


class GradientNorm(Attack):
    """
    Compute gradient norm of the model wrt input.
    """

    def __init__(self, model, criterion, **kwargs):
        super().__init__("GradientNorm", model, criterion, whitebox=True)

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        x, y = x.cuda(), y.cuda()
        scores = []
        for x_, y_ in zip(x, y):
            gradients = compute_gradients(self.model, self.criterion, x_.unsqueeze(0), y_.unsqueeze(0))
            # Get norm of gradient
            norm = np.linalg.norm(gradients)
            scores.append(-norm)
        return np.array(scores)
