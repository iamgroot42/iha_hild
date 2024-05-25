import numpy as np
import torch as ch


def compute_gradients(model, criterion, a, b, pick_specific_layer: int = None) -> np.ndarray:
    """
    Compute gradients of model given datapoint (a, b)
    """
    model.zero_grad()
    pred = model(a)
    if pred.shape[1] == 1:
        loss = criterion(pred.squeeze(1), b.float()).mean()
    else:
        loss = criterion(pred, b).mean()
    loss.backward()
    grads = []
    total_layers_avail = 0
    for i, p in enumerate(model.parameters()):
        total_layers_avail += 1
        if pick_specific_layer is not None and i != pick_specific_layer:
            continue
        grads.extend(list(p.grad.detach().cpu().numpy().flatten()))
    if len(grads) == 0:
        raise ValueError(
            f"No gradients computed! Was pick_specific_layer set correctly? Total {total_layers_avail} layers available"
        )
    model.zero_grad()
    grads = np.array(grads)
    return grads


def compute_scaled_logit(model, x, y, mse: bool = False) -> np.ndarray:
    """
    Compute scaled logit of model given datapoints
    """
    model.cuda()
    oprediction = model(x).detach()
    model.cpu()
    eps = 1e-45

    if mse:
        # oprediction will include sigmoid, so reverse-engineer that first
        logit = ch.log(oprediction + eps) - ch.log(1 - oprediction + eps)
        return logit.cpu().numpy()

    # For numerical stability
    predictions = oprediction - ch.max(oprediction, dim=1, keepdim=True)[0]
    predictions = ch.exp(predictions)
    predictions = predictions / ch.sum(predictions, dim=1, keepdim=True)

    y_true = predictions[range(predictions.shape[0]), y]
    y_wrong = 1 - y_true

    logit = ch.log(y_true + eps) - ch.log(y_wrong + eps)
    return logit.cpu().numpy()


# Create a CustomSampler that always includes some point X in batch, and samples remaining points from other_data_source
class SpecificPointIncludedLoader:
    def __init__(self, given_loader, interest_point, num_batches: int):
        self.given_loader = given_loader
        self.interest_point = interest_point
        self.num_batches = num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            x, y = next(iter(self.given_loader))
            # x, y are torch tensors
            # append x, y into these tensors
            x = ch.cat([x, self.interest_point[0].unsqueeze(0)])
            y = ch.cat([y, ch.tensor([self.interest_point[1]])])
            batch = (x, y)
            yield batch

    def __len__(self):
        return self.num_batches
