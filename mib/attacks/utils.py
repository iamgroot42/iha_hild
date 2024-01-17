"""
    Helper functions for attacks
"""
from mib.attacks.loss import LOSS, Logit
from mib.attacks.lira import LiRAOffline
from mib.attacks.unlearn import Unlearning
from mib.attacks.gradient import GradientNorm
from mib.attacks.adv import SDFNorm


ATTACK_MAPPING = {
    "LOSS": LOSS,
    "Logit": Logit,
    "LiRAOffline": LiRAOffline,
    "UnlearnGradNorm": Unlearning,
    "GradNorm": GradientNorm,
    "SDFNorm": SDFNorm,
}


def get_attack(name: str):
    if name not in ATTACK_MAPPING:
        raise ValueError(f"Attack {name} not found.")
    model_class = ATTACK_MAPPING[name]
    return model_class
