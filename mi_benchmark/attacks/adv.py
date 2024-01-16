"""
    Use adversarial attacks to derive signal for membership.
"""
import numpy as np
import torch as ch

from bbeval.attacker.transfer_methods._manipulate_input import (
    clip_by_tensor,
    transformation_function,
)
from bbeval.config import AttackerConfig, ExperimentConfig
from bbeval.attacker.transfer_methods.SMIMIFGSM import SMIMIFGSM

from mi_benchmark.train import get_model_and_criterion, load_data
from tqdm import tqdm

from autoattack import AutoAttack
from superdeepfool.attacks.SuperDeepFool import SuperDeepFool

from mi_benchmark.attacks.base import Attack


class SDFNorm(Attack):
    """
    Measure perturbation norm using Super DeepFool as a signal
    """

    def __init__(self, model, use_actual_target: bool = True):
        super().__init__("SDF-Norm", model, whitebox=True)
        self.use_actual_target = use_actual_target

    def compute_scores(self, x, y) -> np.ndarray:
        attack = SuperDeepFool(
            self.model,
            steps=1000,
            overshoot=0.01,
            search_iter=50,
            number_of_samples=1,
            l_norm="L2",
        )
        norms = []
        for x_, y_ in tqdm(zip(x, y), desc="Computing SDF signals", total=len(x)):
            norms.append(self._compute_score(attack, x_, y_))
        return np.array(norms)

    def _compute_score(self, attack, x_, y) -> float:
        # Compute loss for x as well
        x_ = x_.unsqueeze(0)
        pred = self.model(x_)
        if self.use_actual_target:
            y_t = y[0]
        else:
            y_t = pred[0].argmax()
        y_t = y_t.unsqueeze(0)
        adv = attack(x_, y_t, verbose=False)
        norm_diff = ch.norm(adv - x_, p=2).item()
        return norm_diff


class AutoAttack(Attack):
    """
    Measure perturbation norm using AutoAttack.
    """

    def __init__(self, model, norm="Linf"):
        super().__init__("SDF-Norm", model, whitebox=True)
        self.norm = norm

    def compute_scores(self, x, y) -> np.ndarray:
        adversary = AutoAttack(self.model, norm=self.norm, eps=1, version="standard")
        norms = []
        for x_, y_ in tqdm(
            zip(x, y), desc="Computing AutoAttack signals", total=len(x)
        ):
            adversary = AutoAttack(self.model, norm="Linf", eps=1, version="standard")
            adv = adversary.run_standard_evaluation(
                x_.unsqueeze(0), y_.unsqueeze(0), bs=1
            )
            diff = ch.norm(adv - x_, p=float(self.norm)).cpu().item()
            norms.append(diff)
        return np.array(norms)
