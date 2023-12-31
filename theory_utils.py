
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
    # Set model to eval mode
    model.eval()

    # Compute gradients of model on all given data
    m1, m2 = ch.zeros_like(population_grads), ch.zeros_like(population_grads)
    batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        flat_grads = ch.cat(
            [p.grad.detach().cpu().flatten() for p in model.parameters()]
        )
        grad_noise = population_grads - flat_grads
        m1 += grad_noise
        m2 += grad_noise**2
        batches += 1

    m1 /= batches
    m2 /= batches
    return m1, m2
