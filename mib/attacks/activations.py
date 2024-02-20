"""
    Attack that looks at activations in a certain layer.
"""

import numpy as np
import torch as ch

from mib.attacks.base import Attack


class Activations(Attack):
    """
    Activation analysis attack
    """

    def __init__(self, model):
        super().__init__("Activations", model, reference_based=False)

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        x = x.cuda()
        # out_models = kwargs.get("out_models", None)
        # if out_models is None:
        # raise ValueError("Reference attack requires out_models to be specified")
        # return compute_ref_score(self.model, x, y, out_models, self.criterion)
        pick_layer = 2 # 2 works best, 4 is good when flipped?, 5 absolutely useless
        acts = self.model(x, layer_readout=pick_layer).detach().cpu().view(len(x), -1).numpy()
        nonzero_acts = np.sum(acts > 0, 1) * 1.
        return nonzero_acts


class ActivationsOffline(Attack):
    """
    Activation analysis attack, normalized with offline models.
    """

    def __init__(self, model):
        super().__init__("ActivationsOffline", model, reference_based=True)

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        x = x.cuda()

        out_models = kwargs.get("out_models", None)
        if out_models is None:
            raise ValueError("ActivationsOffline attack requires out_models to be specified")
        pick_layer = 0

        self.model.cuda()
        acts = self.model(x, layer_readout=pick_layer).detach().cpu().view(len(x), -1).numpy()
        self.model.cpu()

        nonzero_acts = np.sum(acts > 0, 1) * 1.

        out_acts = []
        for out_m in out_models:
            out_m.cuda()
            acts = out_m(x, layer_readout=pick_layer).detach().cpu().view(len(x), -1).numpy()
            out_m.cpu()

            act_ref = np.sum(acts > 0, 1)
            out_acts.append(act_ref)
        out_acts = np.array(out_acts, dtype=np.float64)

        nonzero_acts -= np.mean(out_acts, 0)
        return nonzero_acts
