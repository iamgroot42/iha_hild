import torch as ch
import numpy as np
from torchvision import transforms


@ch.no_grad()
def get_bn_stats(model, loader, flipped_also: bool = False):
    """
    Extract bathnorm statistics from model, followed by computing distances based on this statistics
    for all points present in given loader.
    """
    flip_transform = transforms.RandomHorizontalFlip()
    model.cpu()
    all_features = None
    stats = []
    for x, y in loader:
        _, (features, bnorm_means, bnorm_vars) = model(x, pre_bn_layers=True)
        if all_features is None:
            # One array per layer-statistics
            all_features = [[] for _ in range(len(features))]

        if flipped_also:
            x_flipped = flip_transform(x)
            _, (features_flipped, _, _) = model(x_flipped, pre_bn_layers=True)

        for id, f in enumerate(features):
            reduced = f.detach().mean(dim=2).mean(dim=2)
            if flipped_also:
                reduced_flipped = features_flipped[id].detach().mean(dim=2).mean(dim=2)
                reduced = (reduced + reduced_flipped) / 2
            all_features[id].append(reduced)

    for i, f in enumerate(all_features):
        all_features[i] = ch.cat(f, dim=0)

    for i, (mean, var) in enumerate(zip(bnorm_means, bnorm_vars)):
        diff = all_features[i] - mean
        # diff /= ch.sqrt(var + 1e-5)
        stat = ch.norm(diff, dim=1, p=2).numpy()
        stats.append(stat)
    stats = np.array(stats).T
    return stats
