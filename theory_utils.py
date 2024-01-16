
import torch as ch


def compute_gradients_on_data(model, loader, criterion, device):
    # Set model to eval mode
    model.eval()

    # Compute gradients of model on all given data
    running_gradients = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        flat_grads = ch.cat(
            [p.grad.detach().cpu().flatten() for p in model.parameters()]
        )
        running_gradients.append(flat_grads)
    # Compute mean gradient (across data, per param)
    mean_grads = ch.stack(running_gradients).mean(dim=0)

    # Zero out model gradients now
    model.zero_grad()

    return mean_grads


def compute_grad_noise_statistics(model, population_grads, loader, criterion, device):
    traindata_grads = compute_gradients_on_data(model, loader, criterion, device)

    grad_noise = traindata_grads - population_grads
    
    m1 = grad_noise
    m2 = grad_noise ** 2

    return m1, m2
