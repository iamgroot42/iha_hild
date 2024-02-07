"""
    Use adversarial attacks to derive signal for membership.
"""
import numpy as np
import torch as ch

from tqdm import tqdm

from autoattack import AutoAttack
from superdeepfool.attacks.SuperDeepFool import SuperDeepFool

from mib.attacks.base import Attack


class SDFNorm(Attack):
    """
    Measure perturbation norm using Super DeepFool as a signal
    """

    def __init__(self, model):
        super().__init__("SDFNorm", model, whitebox=True)

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        x, y = x.cuda(), y.cuda()
        steps = kwargs.get("steps", 1000)
        search_iter = kwargs.get("search_iter", 50)
        use_actual_target = kwargs.get("use_actual_target", True)

        attack = SuperDeepFool(
            self.model,
            steps=steps,
            overshoot=0.01,
            search_iter=search_iter,
            number_of_samples=1,
            l_norm="L2",
        )
        norms = []
        for x_, y_ in zip(x, y):
            norms.append(self._compute_score(attack, x_.unsqueeze(0), y_.unsqueeze(0), use_actual_target))
        return np.array(norms)

    def _compute_score(self, attack, x_, y, use_actual_target: bool) -> float:
        pred = self.model(x_)
        if use_actual_target:
            y_t = y
        else:
            y_t = pred.argmax(1)
        adv = attack(x_, y_t, verbose=False)
        norm_diff = ch.norm(adv - x_, p=2).item()
        return norm_diff


class AutoAttackNorm(Attack):
    """
    Measure perturbation norm using AutoAttack.
    """

    def __init__(self, model):
        super().__init__("SDF-Norm", model, whitebox=True)

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        norm = kwargs.get("norm", "inf")
        x, y = x.cuda(), y.cuda()

        adversary = AutoAttack(self.model, norm='L' + str(norm),
                               eps=1, version="standard", verbose=False)
        norms = []
        for x_, y_ in tqdm(
            zip(x, y), desc="Computing AutoAttack signals", total=len(x)
        ):
            adversary = AutoAttack(self.model, norm="Linf", eps=1, version="standard")
            adv = adversary.run_standard_evaluation(
                x_.unsqueeze(0), y_.unsqueeze(0), bs=1
            )
            diff = ch.norm(adv - x_, p=float(norm)).cpu().item()
            norms.append(diff)
        return np.array(norms)
