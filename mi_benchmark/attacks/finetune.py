"""
    Finetuning/Unlearning based attacks.
"""
import numpy as np
from mi_benchmark.attacks.base import Attack
from mi_benchmark.attacks.attack_utils import compute_gradient_norm


class Unlearning(Attack):
    """
    Unlearning attack - records metrics before/after unlearning.
    """

    def __init__(self, model):
        super().__init__("Unlearning", model, whitebox=True)

    def compute_scores(self, x, y) -> np.ndarray:
        scores = []
        for x_, y_ in zip(x, y):
            gradients = compute_gradient_norm(self.model, self.criterion, x_, y_)
            # Get norm of gradient
            norm = np.linalg.norm(gradients)
            scores.append(norm)
        return np.array(scores)
