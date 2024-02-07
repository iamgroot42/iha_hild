import torch as ch


def compute_gradients_on_data(model, loader, criterion, device, population_grads= None):
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
            grad_noise = flat_grads - population_grads
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
