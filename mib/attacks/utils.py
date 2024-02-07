"""
    Helper functions for attacks
"""
from mib.attacks.loss import LOSS, Logit, LOSSSmooth
from mib.attacks.lira import LiRAOffline, LiRAOnline
from mib.attacks.unlearn import Unlearning
from mib.attacks.gradient import GradientNorm
from mib.attacks.adv import SDFNorm, AutoAttackNorm
from mib.attacks.theory import TheoryRef
from mib.attacks.reference import Reference, ReferenceSmooth


ATTACK_MAPPING = {
    "LOSS": LOSS,
    "LOSSSmooth": LOSSSmooth,
    "Logit": Logit,
    "LiRAOffline": LiRAOffline,
    "LiRAOnline": LiRAOnline,
    "UnlearnGradNorm": Unlearning,
    "GradNorm": GradientNorm,
    "SDFNorm": SDFNorm,
    "AutoAttackNorm": AutoAttackNorm,
    "TheoryRef": TheoryRef,
    "Reference": Reference,
    "ReferenceSmooth": ReferenceSmooth,
}


def get_attack(name: str):
    if name not in ATTACK_MAPPING:
        raise ValueError(f"Attack {name} not found.")
    model_class = ATTACK_MAPPING[name]
    return model_class
