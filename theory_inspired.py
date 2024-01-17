"""
    Inference attack inspired by theoretical result around Bayes-optimal inference.
    1. Compute m_1 and m_2, approximations of gradient noist (and geadient noise^2).
    2. Train a few reference models (with all records except the one being tested)
    3. Compute statistics assuming all of these values to come up with scores.
"""
import torch as ch
import numpy as np
from torch.nn import functional as F
import torch.nn as nn

import numpy as np
from tqdm import tqdm

from mib.train import get_model_and_criterion, load_data


def gradient_noise_parameters(model, loader_pop, loader_data):
    """
    Compute expected gradient noise for model training. First computes average gradient over entire population,
    and then computes the expected gradient noise over the population. Returns m_1 and m_2, based on ()^1 and ()^2
    respectively.
    """
    model.eval()
    running_gradients = []
    for batch in loader_pop:
        x, y = batch
        x, y = x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)
        model.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        # Flatten out all gradients (treat as one long list of params)
        flat_grads = ch.cat(
            [p.grad.detach().cpu().flatten() for p in model.parameters()]
        )
        running_gradients.append(flat_grads)

    # Compute mean gradient
    mean_gradient = ch.stack(running_gradients).mean(dim=0)

    # Re-run, with a different loader now
    m_1, m_2 = ch.zeros(mean_gradient.shape), ch.zeros(mean_gradient.shape)
    for batch in loader_data:
        x, y = batch
        x, y = x.to("cuda", non_blocking=True), y.to("cuda", non_blocking=True)
        model.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        # Flatten out all gradients (treat as one long list of params)
        flat_grads = ch.cat(
            [p.grad.detach().cpu().flatten() for p in model.parameters()]
        )
        diff = flat_grads - mean_gradient
        m_1 += diff
        m_2 += diff**2
    m_1 /= len(loader_data)
    m_2 /= len(loader_data)

    m_1 = ch.unsqueeze(m_1, dim=0).cuda()
    m_2 = ch.unsqueeze(m_2, dim=0).cuda()

    return m_1, m_2


def create_phi(m_1, m_2):
    phi = ch.zeros(m_1.shape[0], m_1.shape[0]).cuda()
    for i in range(m_1.shape[0]):
        phi[i, i] = 1 / (m_2[i] - m_1[i] ** 2)
    return phi


def main():
    # Load target model
    model_dict = ch.load("target_model/1.pt")
    # Extract member information and model
    model_weights = model_dict["model"]
    model, _ = get_model_and_criterion("cifar10", device="cpu")
    model.load_state_dict(model_weights, strict=False)
    # Shift to CUDA
    model = model.cuda()
    # Make sure it's on eval model
    model.eval()

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

    # CIFAR
    num_pop_points = 25000
    feature_dim = 16
    num_classes = 10
    num_samples_test = 50

    # Create Subset datasets for members
    member_dset = ch.utils.data.Subset(all_data, train_index)
    # and non- members
    nonmember_indices = np.random.choice(other_indices, num_pop_points, replace=False)
    nonmember_dset = ch.utils.data.Subset(all_data, nonmember_indices)

    # Create loader for population
    loader_pop = ch.utils.data.DataLoader(
        nonmember_dset, batch_size=64, shuffle=True, num_workers=4
    )
    # Create train-specific loader
    loader_data = ch.utils.data.DataLoader(
        member_dset, batch_size=64, shuffle=True, num_workers=4
    )

    # Step 1- estimate m1, m2 for given target model
    m1, m2 = gradient_noise_parameters(model, loader_pop, loader_data)
    # Construct phi matrix
    phi = create_phi(m1, m2)
    


if __name__ == "__main__":
    main()
