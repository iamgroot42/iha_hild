"""
    LiRA, as described in https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9833649
"""
import numpy as np
from typing import List
import scipy
import torch as ch

from mib.attacks.base import Attack
from mib.attacks.attack_utils import compute_scaled_logit


class LiRAOffline(Attack):
    """
    Offline verion of LiRA
    """

    def __init__(self, model, fix_variance: bool = True):
        super().__init__("LiRAOffline", model, reference_based=True)
        self.fix_variance = fix_variance

    @ch.no_grad()
    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        x, y = x.cuda(), y.cuda()
        out_models = kwargs.get("out_models", None)
        if out_models is None:
            raise ValueError("LiRAOffline requires out_models to be specified")
        observed_conf = compute_scaled_logit(self.model, x, y)

        confs = []
        for out_model in out_models:
            out_model.cuda()
            conf = compute_scaled_logit(out_model, x, y)
            out_model.cpu()
            confs.append(conf)
        confs = np.array(confs)

        # Original code uses median instead of mean, so we'll do that too
        mean_out = np.median(confs, 1)

        if self.fix_variance:
            std_out = np.std(confs)
        else:
            std_out = np.std(confs, 1)

        scores = []
        for oc in observed_conf:
            score = scipy.stats.norm.logpdf(oc, mean_out, std_out + 1e-30)
            # scores.append(score.mean(1))
            scores.append(score.mean())
        scores = np.array(scores)
        return scores
