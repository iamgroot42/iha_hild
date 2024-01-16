import torch as ch
from typing import List
import numpy as np
import torch.nn as nn

from distribution_inference.attacks.whitebox.core import BasicDataset
import distribution_inference.datasets.utils as utils
from distribution_inference.config.core import WhiteBoxAttackConfig
from torch.utils.data import DataLoader


def _get_weight_layers(
    model: nn.Module,
    start_n: int = 0,
    first_n: int = None,
    custom_layers: List[int] = None,
    include_all: bool = False,
    is_conv: bool = False,
    transpose_features: bool = True,
    prune_mask=[],
    detach: bool = True,
    track_grad: bool = False,
):
    dims, dim_kernels, weights, biases = [], [], [], []
    i, j = 0, 0

    # Treat 'None' as int
    first_n = np.inf if first_n is None else first_n

    # Sort and store desired layers, if specified
    custom_layers_sorted = sorted(custom_layers) if custom_layers is not None else None

    # Used to keep track of batch-norm layers (and when to skip them)
    is_skip_next_used_for_bn = False

    track = 0
    for name, param in model.named_parameters():
        ### <BN-related logic> ###

        # For now, we ignore batch-norm layers
        if "weight" in name and len(param.shape) == 1:
            is_skip_next_used_for_bn = True
            continue

        # Skip 'bias' of batch-norm layer as well
        if is_skip_next_used_for_bn:
            is_skip_next_used_for_bn = False
            continue
        ### </BN-related logic> ###

        # WEIGHT
        if "weight" in name:
            if track_grad:
                param_data = param
            else:
                param_data = param.data
            if detach:
                param_data = param_data.detach()
            param_data = param_data.cpu()

            # Apply pruning masks if provided
            if len(prune_mask) > 0:
                param_data = param_data * prune_mask[track]
                track += 1

            if transpose_features:
                param_data = param_data.T

            weights.append(param_data)
            if is_conv:
                dims.append(weights[-1].shape[2])
                dim_kernels.append(weights[-1].shape[0] * weights[-1].shape[1])
            else:
                dims.append(weights[-1].shape[0])
        # BIAS
        if "bias" in name:
            if track_grad:
                param_data = param
            else:
                param_data = param.data
            if detach:
                param_data = param_data.detach()
            param_data = param_data.cpu()
            biases.append(ch.unsqueeze(param_data, 0))

        # Assume each layer has weight & bias
        i += 1

        if custom_layers_sorted is None:
            # If requested, start looking from start_n layer
            if (i - 1) // 2 < start_n:
                dims, dim_kernels, weights, biases = [], [], [], []
                continue

            # If requested, look at only first_n layers
            if i // 2 > first_n - 1:
                break
        else:
            # If this layer was not asked for, omit corresponding weights & biases
            if i // 2 != custom_layers_sorted[j // 2]:
                dims = dims[:-1]
                dim_kernels = dim_kernels[:-1]
                weights = weights[:-1]
                biases = biases[:-1]
            else:
                # Specified layer was found, increase count
                j += 1

            # Break if all layers were processed
            if len(custom_layers_sorted) == j // 2:
                break

    if custom_layers_sorted is not None and len(custom_layers_sorted) != j // 2:
        raise ValueError("Custom layers requested do not match actual model")

    if include_all:
        if is_conv:
            middle_dim = weights[-1].shape[3]
        else:
            middle_dim = weights[-1].shape[1]

    cctd = []
    for w, b in zip(weights, biases):
        if is_conv:
            b_exp = b.unsqueeze(0).unsqueeze(0)
            b_exp = b_exp.expand(w.shape[0], w.shape[1], 1, -1)
            combined = ch.cat((w, b_exp), 2).transpose(2, 3)
            combined = combined.view(-1, combined.shape[2], combined.shape[3])
        else:
            combined = ch.cat((w, b), 0).T

        cctd.append(combined)

    if is_conv:
        if include_all:
            return (dims, dim_kernels, middle_dim), cctd
        return (dims, dim_kernels), cctd
    if include_all:
        return (dims, middle_dim), cctd
    return dims, cctd


# Function to extract model parameters
def get_weight_layers(
    model: nn.Module,
    attack_config: WhiteBoxAttackConfig,
    prune_mask=[],
    detach: bool = True,
    track_grad: bool = False,
):
    # Model has convolutional layers
    # Process FC and Conv layers separately
    dims_conv, fvec_conv = _get_weight_layers(
        model.features,
        first_n=attack_config.first_n_conv,
        start_n=attack_config.start_n_conv,
        is_conv=True,
        custom_layers=attack_config.custom_layers_conv,
        transpose_features=model.transpose_features,
        prune_mask=prune_mask,
        include_all=True,
        detach=detach,
        track_grad=track_grad,
    )

    # Some models (relation-net, etc) may not have linear layers
    if model.classifier is None:
        dims_fc, fvec_fc = None, None
    else:
        dims_fc, fvec_fc = _get_weight_layers(
            model.classifier,
            first_n=attack_config.first_n_fc,
            start_n=attack_config.start_n_fc,
            custom_layers=attack_config.custom_layers_fc,
            transpose_features=model.transpose_features,
            prune_mask=prune_mask,
            detach=detach,
            track_grad=track_grad,
        )
    # If PIN requested only FC layers, return only FC layers
    if attack_config.permutation_config:
        if attack_config.permutation_config.focus == "fc":
            feature_vector = fvec_fc
        elif attack_config.permutation_config.focus == "conv":
            feature_vector = fvec_conv
        else:
            feature_vector = fvec_conv + fvec_fc
    dimensions = (dims_conv, dims_fc)

    return dimensions, feature_vector


def wrap_into_loader(
    features_list: List,
    batch_size: int,
    labels_list: List[float] = [0.0, 1.0],
    shuffle: bool = False,
    num_workers: int = 2,
    wrap_with_loader: bool = True,
    epochwise_version: bool = False,
):
    """
    Wrap given features of models from N distributions
    into X and Y, to be used for model training. Use given list of
    labels for each distribution.
    """
    # Special case if epoch-wise version
    if epochwise_version:
        loaders_list = []
        # We want one loader per epoch
        n_epochs = len(features_list[0][0])
        for i in range(n_epochs):
            loaders_list.append(
                wrap_into_loader(
                    [features[:, i] for features in features_list],
                    batch_size,
                    labels_list,
                    shuffle,
                    num_workers,
                    wrap_with_loader,
                    epochwise_version=False,
                )
            )
        return loaders_list

    # Everything else:
    X, Y = [], []
    for features, label in zip(features_list, labels_list):
        X.append(features)
        Y.append([label] * len(features))
    Y = np.concatenate(Y, axis=0)

    # Return in loader form if requested
    if wrap_with_loader:
        X = np.concatenate(X, axis=0)
        loader = covert_data_to_loaders(
            X, Y, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
    else:
        X = np.concatenate(X, axis=0, dtype=object)
        Y = ch.Tensor(Y)
        loader = (X, Y)
    return loader


def covert_data_to_loaders(
    X,
    Y,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 2,
    reduce: bool = False,
):
    """
    Create DataLoaders using given model features and labels.
    Will delete given X, Y vectors to save memory.
    """

    # Define collate_fn
    def collate_fn(data):
        features, labels = zip(*data)
        # Combine them per-layer
        x = [[] for _ in range(len(features[0]))]
        for feature in features:
            for i, layer_feature in enumerate(feature):
                x[i].append(layer_feature)

        x = [ch.stack(x_, 0) for x_ in x]
        if reduce:
            x = [x_.view(-1, x_.shape[-1]) for x_ in x]
        y = ch.tensor(labels).float()

        return x, y

    # Create your own dataset
    ds = BasicDataset(X, Y)

    # Get loader using given dataset
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=utils.worker_init_fn,
        prefetch_factor=2,
    )
    return loader
