"""
    Attack based on theoretical analysis of SGD.
"""
import numpy as np
import torch as ch
import torch.nn as nn

from mib.attacks.base import Attack


class TheoryRef(Attack):
    """
    Attack that computes trace to normalize score and uses those for attack.
    Uses reference models to compute normalization
    """

    def __init__(self, model):
        super().__init__("TheoryRef", model, reference_based=True)

    def register_trace(self, trace):
        self.trace = trace

    @ch.no_grad()
    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        x, y = x.cuda(), y.cuda()
        out_models = kwargs.get("out_models", None)
        out_traces = kwargs.get("out_traces", None)

        if out_models is None:
            raise ValueError("TheoryRef requires out_models to be specified")
        if out_traces is None:
            raise ValueError("TheoryRef requires out_traces to be specified")

        losses = []
        for out_model, out_trace in zip(out_models, out_traces):
            out_model.cuda()
            loss = self.criterion(out_model(x).detach(), y).cpu().numpy()
            out_model.cpu()
            scaled_loss = loss * out_trace
            losses.append(scaled_loss)
        ref_factor = np.mean(losses, 0)
        if len(losses) == 0:
            ref_factor = 0

        self.model.cuda()
        current_score = (
            self.criterion(self.model(x).detach(), y).cpu().numpy() * self.trace
        )
        self.model.cpu()

        scores = -(current_score - ref_factor)
        return scores


def compute_gradients_on_data(model, loader, criterion, device, population_grads=None):
    # Set model to eval mode
    model.eval()

    # Compute gradients of model on all given data
    running_gradients = []
    m1, m2 = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        flat_grads = ch.cat(
            [p.grad.detach().cpu().flatten() for p in model.parameters()]
        )
        if population_grads is not None:
            grad_noise = population_grads - flat_grads
            m1.append(grad_noise)
            m2.append(grad_noise**2)
        else:
            running_grad = flat_grads
            running_gradients.append(running_grad)

    # Zero out model gradients now
    model.zero_grad()

    if population_grads is not None:
        m1 = ch.stack(m1).mean(dim=0)
        m2 = ch.stack(m2).mean(dim=0)
        return m1, m2

    # Compute mean gradient (across data, per param)
    mean_grads = ch.stack(running_gradients).mean(dim=0)
    return mean_grads


def compute_grad_noise_statistics(model, population_grads, loader, criterion, device):
    m1, m2 = compute_gradients_on_data(
        model, loader, criterion, device, population_grads
    )

    return m1, m2


def compute_trace(model, train_loader, test_loader,
                  device="cuda",
                  shift_model: bool = True,
                  get_m1_m2: bool = False):
    if shift_model:
        model.to(device)
    criterion = nn.CrossEntropyLoss()
    population_grads = compute_gradients_on_data(model, test_loader, criterion, device)

    m1, m2 = compute_gradients_on_data(
        model, train_loader, criterion, device, population_grads
    )
    if shift_model:
        model.cpu()

    # Scale with learning rate
    eps = 1e-18
    trace_value = ch.sum(1 / (m2 - m1**2 + eps))
    # Scale with dimensionality
    trace_value /= len(m1)
    if get_m1_m2:
        return trace_value.cpu().numpy(), m1.detach(), m2.detach()
    return trace_value.cpu().numpy()