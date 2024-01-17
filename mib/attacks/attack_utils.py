import numpy as np
import torch as ch


def compute_gradients(model, criterion, a, b) -> np.ndarray:
    """
    Compute gradients of model given datapoint (a, b)
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
