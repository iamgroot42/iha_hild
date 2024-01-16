import numpy as np
import torch as ch


def compute_gradient_norm(model, criterion, a, b) -> np.ndarray:
    """
    Compute gradient norm of model given datapoint (a, b)
    """
    model.zero_grad()
    pred = model(a)
    loss = criterion(pred, b).mean()
    loss.backward()
    grads = []
    for p in model.parameters():
        grads.extend(list(p.grad.detach().cpu().numpy().flatten()))
    model.zero_grad()
    grads = np.array(grads)
    return grads


def compute_scaled_logit(model, x, y) -> np.ndarray:
    """
    Compute scaled logit of model given datapoints
    """
    model.cuda()
    oprediction = model(x).detach()
    model.cpu()
    # For numerical stability
    predictions = oprediction - ch.max(oprediction, dim=1, keepdim=True)[0]
    predictions = ch.exp(predictions)
    predictions = predictions / ch.sum(predictions, dim=1, keepdim=True)

    y_true = predictions[range(predictions.shape[0]), y]
    y_wrong = 1 - y_true

    eps = 1e-45
    logit = ch.log(y_true + eps) - ch.log(y_wrong + eps)
    return logit.cpu().numpy()
