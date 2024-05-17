"""Base class for attacks."""
import torch as ch
import numpy as np


class Attack(object):
    """
    Base class for all attacks. Need to check level of acces required by attack,
    whether reference models are needed, mode of attack (online or offline).
    Args:
        name (str): Name of the attack
        model (nn.Module): Model to attack
    """

    def __init__(
        self,
        name: str,
        model,
        criterion,
        whitebox: bool = False,
        reference_based: bool = False,
        requires_trace: bool = False,
    ):
        self.name = name
        self.model = model
        self.criterion = criterion
        self.whitebox = whitebox
        self.reference_based = reference_based
        self.requires_trace = requires_trace

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        """
        Compute the score of the attack. Must be implemented by child class.
        """
        raise NotImplementedError("Attack not implemented")
