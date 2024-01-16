"""
    Use the neuron-attribution work from https://arxiv.org/pdf/2307.09542.pdf and
    repurpose as a white-box attack.
"""
import torch as ch
import torch.nn as nn
import copy
import numpy as np
from tqdm import tqdm

from mi_benchmark.train import get_dataset_subset, load_data, get_model_and_criterion


def identify_neurons(model, data_batch, num_random: int = 5, noise_scale: float = 0.03):
    """
    data_batch: batch of data to use for identifying neurons. Only first element is used, other is used as "other" data
    num_random: number of times random noise is added to data before aggregating model outputs
    noise_scale: scale of random noise to add to inputs
    """
    criterion = nn.CrossEntropyLoss(reduction="none")

    # Note that the attack right now is very specific to relation-network and subject-level MI
    @ch.no_grad()
    def noise_at_scale(x):
        noise = ch.normal(0, 1, size=x.shape)
        noise *= noise_scale * x.norm() / noise.norm()
        return x + noise

    # Take note of predictions at start
    pred_start = ch.argmax(model(data_batch[0][:1].cuda()), 1).item()

    def get_noisy_preds(data):
        preds_avg = []
        for _ in range(num_random):
            data_noisy = noise_at_scale(data[0])
            preds = model_(data_noisy.cuda())
            preds_avg.append(preds)
        preds_avg = ch.stack(preds_avg, dim=0).mean(dim=0)
        return preds_avg, data[1]

    # Make a copy of model (we will modify weights)
    model_ = copy.deepcopy(model)
    pred_flipped = False
    names, locs = [], []
    max_lim = 0
    found_max_lim = False
    while not pred_flipped:
        # Start with clean slate
        model_.zero_grad()

        x, y = get_noisy_preds(data_batch)
        x, y = x.cuda(), y.cuda()

        # Compute target and others loss
        loss = criterion(x, y.long())
        loss[0] *= -1 * len(x)
        loss = loss.mean()

        # Compute gradients with this loss
        loss.backward()

        # Look at gradients
        max_name, max_loc, max_grad = None, None, 0
        for name, param in model_.named_parameters():
            if "weight" not in name or param.grad is None:
                continue

            # Get gradients for this layer
            grad_eff = param.data * param.grad.detach()

            # Handle conv case
            if len(param.data.shape) == 4:
                grad_eff = grad_eff.sum(dim=(1, 2, 3))

            #  Helps with dynamic computation of # of parameters
            if not found_max_lim:
                max_lim += grad_eff.shape[0]

            # Pick arg-max channel (neuron) that maximizes this loss
            if ch.max(grad_eff) > max_grad:
                max_grad = ch.max(grad_eff)
                max_name = name
                max_loc = unravel_index(ch.argmax(grad_eff), grad_eff.shape)

        if not found_max_lim:
            found_max_lim = True  # Done calculating max_lim
            max_lim //= 5  # We do not want to zero out more than 20% of the network!

        # Keep track of neurons
        names.append(max_name)
        locs.append(max_loc)

        # Zero out specified weight
        with ch.no_grad():
            # Zero out weight of parameter with name max_name at location max_loc
            # Fetch directly via model named parameters
            for name, param in model_.named_parameters():
                if name == max_name:
                    param.data[max_loc] = 0
                    break

            # Check if preds flipped
            pred_flipped = (
                pred_start != ch.argmax(model_(data_batch[0][:1].cuda()), 1).item()
            )

            # Break if preds flipped
            if pred_flipped:
                break

        # Clear CUDA cache
        ch.cuda.empty_cache()

        # Break if reached max_lim capacity
        if len(names) >= max_lim:
            break

    return names, locs, pred_flipped


def unravel_index(index, shape):
    # torch.argmax returns index for a flattened tensor. to be able to index it later on we need to unravel it.
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


if __name__ == "__main__":
    # Load target model
    model_dict = ch.load("target_model/1.pt")
    # Extract member information and model
    model_weights = model_dict["model"]
    model, _ = get_model_and_criterion("cifar10", device="cuda")
    model.load_state_dict(model_weights, strict=False)

    train_index, test_index = model_dict["train_index"], model_dict["test_index"]

    # Get data
    all_data = load_data(None, None)
    # Get indices out of range(len(all_data)) that are not train_index or test_index
    other_indices = np.array(
        [
            i
            for i in range(len(all_data))
            if i not in train_index and i not in test_index
        ]
    )
    train_dset_x, train_dset_y = get_dataset_subset(all_data, train_index)
    test_dset_x, test_dset_y = get_dataset_subset(all_data, test_index)
    other_dset_x, other_dset_y = get_dataset_subset(all_data, other_indices)

    batch_size = 32
    n_points_test = 50

    def collect_signals(data_source_x, data_source_y):
        signals = []
        for i in tqdm(range(n_points_test)):
            # Pick ith member fro train_dset
            x, y = data_source_x[i], data_source_y[i]
            # Pick batch_size-1 random points from other_dset
            random_indices = np.random.choice(
                len(other_dset_x), batch_size - 1, replace=False
            )
            other_x, other_y = (
                other_dset_x[random_indices],
                other_dset_y[random_indices],
            )
            # Combine these two
            X = ch.cat((ch.unsqueeze(x, 0), other_x), 0)
            Y = ch.cat((ch.unsqueeze(y, 0), other_y), 0)

            ret_obj = identify_neurons(model, (X, Y))
            signal = len(ret_obj[0])
            signals.append(signal)

        print(signals)

        return np.array(signals)

    # Signals for non-members
    signals_nonmem = collect_signals(test_dset_x, test_dset_y)
    # Signals for members
    signals_mem = collect_signals(train_dset_x, train_dset_y)

    # Prepare data for saving
    signals = np.concatenate((signals_nonmem, signals_mem))
    labels = np.concatenate((np.zeros(len(signals) // 2), np.ones(len(signals) // 2)))

    # Save both these arrays
    np.save("signals.npy", {
        "signals": signals,
        "labels": labels
    })
