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
            pr_out = scipy.stats.norm.logpdf(oc, mean_out, std_out + 1e-30)
            score = pr_out
            # scores.append(score.mean(1))
            scores.append(score.mean())
        scores = np.array(scores)
        return -scores


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
        if out_models is None or in_models is None:
            raise ValueError(
                "LiRAOnline requires out_models and in_models to be specified"
            )
        observed_conf = compute_scaled_logit(self.model, x, y)

        # Out-model confidence values
        confs_out = []
        for out_model in out_models:
            out_model.cuda()
            conf = compute_scaled_logit(out_model, x, y)
            out_model.cpu()
            confs_out.append(conf)
        confs_out = np.array(confs_out)

        # In-model confidence values
        confs_in = []
        for in_model in in_models:
            in_model.cuda()
            conf = compute_scaled_logit(in_model, x, y)
            in_model.cpu()
            confs_in.append(conf)
        confs_in = np.array(confs_in)

        # Enforce same number of in and out models
        min_both = min(len(confs_in), len(confs_out))
        confs_in = confs_in[:min_both]
        confs_out = confs_out[:min_both]

        # Original code uses median instead of mean, so we'll do that too
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
            scores.append(score.mean())
        scores = np.array(scores)
        return scores


class LiRAOnlineSmooth(Attack):
    """
        Version of LiRA-Online that uses label-smoothing in loss computation.
    """
    def __init__(self, model, fix_variance: bool = True):
        super().__init__(
            "LiRAOnlineSmooth", model, reference_based=True, label_smoothing=5e-4
        )
        self.fix_variance = fix_variance

    def _compute_score(self, model, x, y):
        # Equivalent to log(p/1-p) where p is the probability of the true class
        model.cuda()
        loss = self.criterion(model(x.cuda()).detach(), y.cuda())
        model.cpu()
        eps = 1e-45
        score = -ch.log(ch.exp(loss + eps) - 1)
        return score.cpu().numpy()

    @ch.no_grad()
    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        x, y = x.cuda(), y.cuda()
        out_models = kwargs.get("out_models", None)
        in_models = kwargs.get("in_models", None)
        if out_models is None or in_models is None:
            raise ValueError(
                f"{self.name} requires out_models and in_models to be specified"
            )
        observed_conf = self._compute_score(self.model, x, y)

        # Out-model confidence values
        confs_out = []
        for out_model in out_models:
            out_model.cuda()
            conf = self._compute_score(out_model, x, y)
            out_model.cpu()
            confs_out.append(conf)
        confs_out = np.array(confs_out)

        # In-model confidence values
        confs_in = []
        for in_model in in_models:
            in_model.cuda()
            conf = self._compute_score(in_model, x, y)
            in_model.cpu()
            confs_in.append(conf)
        confs_in = np.array(confs_in)

        # Enforce same number of in and out models
        min_both = min(len(confs_in), len(confs_out))
        confs_in = confs_in[:min_both]
        confs_out = confs_out[:min_both]

        # Original code uses median instead of mean, so we'll do that too
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
            scores.append(score.mean())
        scores = np.array(scores)
        return scores
