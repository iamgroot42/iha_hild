"""
    Helper functions for attacks
"""
from mi_benchmark.attacks.loss import LOSS, Logit
from mi_benchmark.attacks.lira import LiRAOffline


ATTACK_MAPPING = {
    "LOSS": LOSS,
    "Logit": Logit,
    "LiRAOffline": LiRAOffline
}


def get_attack(name: str):
    if name not in ATTACK_MAPPING:
        raise ValueError(f"Attack {name} not found.")
    model_class = ATTACK_MAPPING[name]
    return model_class
