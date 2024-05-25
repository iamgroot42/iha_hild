"""
    RMIA, as described in https://arxiv.org/pdf/2312.03262.pdf
"""

import numpy as np
from typing import List
import scipy
import torch as ch

from mib.attacks.base import Attack
from mib.attacks.attack_utils import compute_scaled_logit


class LiRAOnline(Attack):
    """
    Online verion of LiRA
    """

    def __init__(self, model, fix_variance: bool = True):
        super().__init__("LiRAOnline", model, reference_based=True)
        self.fix_variance = fix_variance

    @ch.no_grad()
    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        x, y = x.cuda(), y.cuda()
        out_models = kwargs.get("out_models", None)
        in_models = kwargs.get("in_models", None)
        x_aug = kwargs.get("x_aug", None)
        if out_models is None or in_models is None:
            raise ValueError(
                "LiRAOnline requires out_models and in_models to be specified"
            )
        observed_conf = compute_scaled_logit(self.model, x, y)

        # if specified, also use augmented data
        x_ref_use, y_ref_use = x, y
        if x_aug is not None:
            x_ref_use = x_aug.cuda()
            # Make y_ref_use the same as y, but repeated for each augmentation
            y_ref_use = y.repeat(len(x_ref_use))

        # Out-model confidence values
        confs_out = []
        for out_model in out_models:
            out_model.cuda()
            conf = compute_scaled_logit(out_model, x_ref_use, y_ref_use)
            out_model.cpu()
            confs_out.append(conf)
        confs_out = np.array(confs_out)

        # In-model confidence values
        confs_in = []
        for in_model in in_models:
            in_model.cuda()
            conf = compute_scaled_logit(in_model, x_ref_use, y_ref_use)
            in_model.cpu()
            confs_in.append(conf)
        confs_in = np.array(confs_in)

        # Enforce same number of in and out models
        min_both = min(len(confs_in), len(confs_out))
        confs_in = confs_in[:min_both]
        confs_out = confs_out[:min_both]

        # Original code uses median instead of mean, so we'll do that too
        # MEDIAN HERE IS FOR AUGMENTATIONS, NOT REF MODELS!
        mean_in = np.median(confs_in, 1)
        mean_out = np.median(confs_out, 1)

        if self.fix_variance:
            std_in = np.std(confs_in)
            std_out = np.std(confs_out)
        else:
            std_in = np.std(confs_in, 1)
            std_out = np.std(confs_out, 1)

        scores = []
        for oc in observed_conf:
            pr_in = scipy.stats.norm.logpdf(oc, mean_in, std_in + 1e-30)
            pr_out = scipy.stats.norm.logpdf(oc, mean_out, std_out + 1e-30)
            score = pr_in - pr_out
            # scores.append(score.mean(1))
            # AND THE MEAN HERE IS ALSO ACROSS AUGMENTATIONS. For no augmentations, keep last dim as (1) to make .mean(1) relevant
            scores.append(score.mean())
        scores = np.array(scores)
        return scores
