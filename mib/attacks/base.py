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
        device: str,
        whitebox: bool = False,
        reference_based: bool = False,
        requires_trace: bool = False,
        uses_hessian: bool = False,
    ):
        self.name = name
        self.model = model
        self.criterion = criterion
        self.whitebox = whitebox
        self.reference_based = reference_based
        self.requires_trace = requires_trace
        self.uses_hessian = uses_hessian
        self.device = device

    def get_hessian(self):
        if not self.uses_hessian:
            raise ValueError("Hessian is not used by this attack")
        return self.hessian

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        """
        Compute the score of the attack. Must be implemented by child class.
        """
        raise NotImplementedError("Attack not implemented")
