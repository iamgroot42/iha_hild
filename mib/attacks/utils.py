"""
    Helper functions for attacks
"""
from mib.attacks.loss import LOSS, Logit, LOSSSmooth
from mib.attacks.lira import LiRAOffline, LiRAOnline, LiRAOnlineSmooth
from mib.attacks.gradient import GradientNorm
from mib.attacks.reference import Reference, ReferenceSmooth, ReferenceAlex
from mib.attacks.theory_new import ProperTheoryRef
from mib.attacks.sif import SIF


ATTACK_MAPPING = {
    "LOSS": LOSS,
    "LOSSSmooth": LOSSSmooth,
    "Logit": Logit,
    "LiRAOffline": LiRAOffline,
    "LiRAOnline": LiRAOnline,
    "LiRAOnlineSmooth": LiRAOnlineSmooth,
    "GradNorm": GradientNorm,
    "Reference": Reference,
    "ReferenceSmooth": ReferenceSmooth,
    "ReferenceAlex": ReferenceAlex,
    "ProperTheoryRef": ProperTheoryRef,
    "SIF": SIF,
}


def get_attack(name: str):
    if name not in ATTACK_MAPPING:
        raise ValueError(f"Attack {name} not found.")
    model_class = ATTACK_MAPPING[name]
    return model_class
