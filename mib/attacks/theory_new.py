"""
    Train a m-layer MLP with multivariate Gaussian distribution as data.
    Allows to work in simplistic setting and have potentially closed-form solutions for these attacks.
"""

import numpy as np
import torch as ch

from mib.attacks.base import Attack
from torch.autograd import functional as F
from torch.autograd import grad
from torch.func import functional_call
from torch_influence import BaseObjective
from torch_influence import AutogradInfluenceModule, LiSSAInfluenceModule, HVPModule


class ProperTheoryRef(Attack):
    """
    I1 + I3 based term that makes assumption about relationship between I2 and I3.
    """
    def __init__(self, model, criterion, all_train_loader, approximate: bool = True):
        super().__init__("ProperTheoryRef", model, criterion, reference_based=False, requires_trace=False, whitebox=True)

        self.approximate = approximate
        self.model.cuda()

        self.all_data_grad = self.collect_grad_on_all_data(all_train_loader)

        # Exact Hessian
        if self.approximate:
            class MyObjective(BaseObjective):
                def train_outputs(self, model, batch):
                    return model(batch[0])

                def train_loss_on_outputs(self, outputs, batch):
                    if outputs.shape[1] == 1:
                        return criterion(outputs.squeeze(1), batch[1].float())  # mean reduction required
                    else:
                        return criterion(outputs, batch[1])  # mean reduction required

                def train_regularization(self, params):
                    return 0

                def test_loss(self, model, params, batch):
                    output = model(batch[0])
                    # no regularization in test loss
                    if output.shape[1] == 1:
                        return criterion(output.squeeze(1), batch[1].float())
                    else:
                        return criterion(output, batch[1])
            
            # LiSSA - 40s/iteration
            # GC - 38s/iteration
            self.ihvp_module = LiSSAInfluenceModule(
                model=model,
                objective=MyObjective(),
                train_loader=all_train_loader,
                test_loader=None,
                device="cuda",
                repeat=2,
                depth=500,  # 5000 for MLP and Transformer, 10000 for CNN
                scale=25,
                damp=1e-2,
            )
        else:
            exact_H = compute_hessian(model, all_train_loader, self.criterion, device="cuda")
            exact_H_touse = exact_H.cpu().clone().detach()

            # Damping
            damping = 1e-1
            L, Q = ch.linalg.eigh(exact_H_touse)
            L += damping
            self.H_inverse = Q @ ch.diag(1 / L) @ Q.T

    def collect_grad_on_all_data(self, loader):
        cumulative_gradients = None
        for x, y in loader:
            # Zero-out accumulation
            self.model.zero_grad()
            # Compute gradients
            x, y = x.cuda(), y.cuda()
            loss = self.criterion(self.model(x).squeeze(), y.float()) * len(x)
            loss.backward()
            flat_grad = []
            for p in self.model.parameters():
                flat_grad.append(p.grad.detach().view(-1))
            # Flatten out gradients
            flat_grad = ch.cat(flat_grad)
            # Accumulate in higher precision
            if cumulative_gradients is None:
                cumulative_gradients = ch.zeros_like(flat_grad, dtype=ch.float64)
            cumulative_gradients += flat_grad
        self.model.zero_grad()
        cumulative_gradients /= len(loader.dataset)
        # TODO: Shift all to float64 precision
        return cumulative_gradients.float()

    def get_specific_grad(self, point_x, point_y):
        self.model.zero_grad()
        logits = self.model(point_x.cuda())
        loss = self.criterion(logits, point_y.unsqueeze(0).float().cuda())
        ret_loss = loss.item()
        loss.backward()
        flat_grad = []
        for p in self.model.parameters():
            flat_grad.append(p.grad.detach().view(-1))
        flat_grad = ch.cat(flat_grad)
        self.model.zero_grad()
        return flat_grad, ret_loss

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        x, y = x.cuda(), y.cuda()
        learning_rate = kwargs.get("learning_rate", None)
        num_samples = kwargs.get("num_samples", None)
        is_train = kwargs.get("is_train", None)

        if is_train is None:
            raise ValueError("ProperTheoryRef requires is_train to be specified (to compute L0 properly)")
        if learning_rate is None:
            raise ValueError("ProperTheoryRef requires knowledge of learning_rate")
        if num_samples is None:
            raise ValueError("ProperTheoryRef requires knowledge of num_samples")

        # Factor out S/(2L*) parts out of both terms as common
        grad, ret_loss = self.get_specific_grad(x, y)
        I1 = ret_loss

        if is_train:
            # Trick to skip computing L0 for all records. Compute L1 (across all data), and then directly calculatee L0 using current grad
            # We are passing 'is_train' flag here but this is not cheating- could be replaced with repeated L0 computation, but would be unnecessarily expensive
            all_other_data_grad = (self.all_data_grad * num_samples - grad) / (num_samples - 1)
        else:
            all_other_data_grad = self.all_data_grad

        if self.approximate:
            datapoint_ihvp = self.ihvp_module.inverse_hvp(grad)
            ihvp_alldata   = self.ihvp_module.inverse_hvp(all_other_data_grad)
        else:
            datapoint_ihvp = (self.H_inverse @ grad.cpu()).cuda()
            ihvp_alldata   = (self.H_inverse @ all_other_data_grad.cpu()).cuda()

        I2 = ch.dot(datapoint_ihvp, datapoint_ihvp).cpu().item() / num_samples
        I3 = ch.dot(ihvp_alldata, datapoint_ihvp).cpu().item() * 2

        I2 /= learning_rate
        I3 /= learning_rate

        mi_score = I1 - (I2 + I3)
        return mi_score / num_samples


def fast_ihvp(model, vec, loader, criterion, device: str = "cpu"):
    """
        Use LiSSA to compute HVP for a given model and dataloader
    """
    class MyObjective(BaseObjective):
        def train_outputs(self, model, batch):
            return model(batch[0])

        def train_loss_on_outputs(self, outputs, batch):
            return criterion(outputs, batch[1])  # mean reduction required

        def train_regularization(self, params):
            return 0

        def test_loss(self, model, params, batch):
            return criterion(
                model(batch[0]), batch[1]
            )  # no regularization in test loss

    module = LiSSAInfluenceModule(
        model=model,
        objective=MyObjective(),
        train_loader=loader,
        test_loader=None,
        device=device,
        damp=0,
        repeat=20,
        depth=100, #5000 for MLP and Transformer, 10000 for CNN
        scale=50 # test in {10, 25, 50, 100, 150, 200, 250, 300, 400, 500} for convergence
    )

    # Get projection of vec onto inverse-Hessian
    ihvp = module.inverse_hvp(vec)

    """
    hvp_module = HVPModule(model, MyObjective(), loader, device="cuda")
    ihvp = hso_ihvp(
        vec,
        hvp_module,
        acceleration_order=10,
        num_update_steps=30,
    )
    """

    return ihvp


def compute_hessian(model, loader, criterion, device: str = "cpu"):
    """
    Compute Hessian at given point
    """
    class MyObjective(BaseObjective):
        def train_outputs(self, model, batch):
            return model(batch[0])

        def train_loss_on_outputs(self, outputs, batch):
            if outputs.shape[1] == 1:
                return criterion(outputs.squeeze(1), batch[1].float())  # mean reduction required
            else:
                return criterion(outputs, batch[1])  # mean reduction required

        def train_regularization(self, params):
            return 0

        def test_loss(self, model, params, batch):
            output = model(batch[0])
            # no regularization in test loss
            if output.shape[1] == 1:
                return criterion(output.squeeze(1), batch[1].float())
            else:
                return criterion(output, batch[1])

    model.zero_grad()

    module = AutogradInfluenceModule(
        model=model,
        objective=MyObjective(),  
        train_loader=loader,
        test_loader=None,
        device=device,
        damp=0,
        store_as_hessian=True
    )

    H = module.get_hessian()
    return H


def compute_epsilon_acceleration(
    source_sequence,
    num_applications: int = 1,
):
    """Compute `num_applications` recursive Shanks transformation of
    `source_sequence` (preferring later elements) using `Samelson` inverse and the
    epsilon-algorithm, with Sablonniere modifier.
    """

    def inverse(vector):
        # Samelson inverse
        return vector / vector.dot(vector)

    epsilon = {}
    for m, source_m in enumerate(source_sequence):
        epsilon[m, 0] = source_m
        epsilon[m + 1, -1] = 0

    s = 1
    m = (len(source_sequence) - 1) - 2 * num_applications
    initial_m = m
    while m < len(source_sequence) - 1:
        while m >= initial_m:
            # Sablonniere modifier
            inverse_scaling = np.floor(s / 2) + 1

            epsilon[m, s] = epsilon[m + 1, s - 2] + inverse_scaling * inverse(
                epsilon[m + 1, s - 1] - epsilon[m, s - 1]
            )
            epsilon.pop((m + 1, s - 2))
            m -= 1
            s += 1
        m += 1
        s -= 1
        epsilon.pop((m, s - 1))
        m = initial_m + s
        s = 1

    return epsilon[initial_m, 2 * num_applications]


@ch.no_grad()
def hso_ihvp(
    vec,
    hvp_module,
    acceleration_order: int = 9,
    initial_scale_factor: float = 1e6,
    num_update_steps: int = 20,
):

    # Detach and clone input
    vector_cache = vec.detach().clone()
    update_sum = vec.detach().clone()
    coefficient_cache = 1

    cached_update_sums = []
    if acceleration_order > 0 and num_update_steps == 2 * acceleration_order + 1:
        cached_update_sums.append(update_sum)

    # Do HessianSeries calculation
    for update_step in range(1, num_update_steps):
        hessian2_vector_cache = hvp_module.hvp(hvp_module.hvp(vector_cache))

        if update_step == 1:
            scale_factor = ch.norm(hessian2_vector_cache, p=2) / ch.norm(vec, p=2)
            scale_factor = max(scale_factor.item(), initial_scale_factor)

        vector_cache = vector_cache - (1 / scale_factor) * hessian2_vector_cache
        coefficient_cache *= (2 * update_step - 1) / (2 * update_step)
        update_sum += coefficient_cache * vector_cache

        if acceleration_order > 0 and update_step >= (
            num_update_steps - 2 * acceleration_order - 1
        ):
            cached_update_sums.append(update_sum.clone())

    update_sum /= np.sqrt(scale_factor)

    # Perform series acceleration (Shanks acceleration)
    if acceleration_order > 0:
        accelerated_sum = compute_epsilon_acceleration(
            cached_update_sums, num_applications=acceleration_order
        )
        accelerated_sum /= np.sqrt(scale_factor)
        return accelerated_sum

    return update_sum


if __name__ == "__main__":
    import torch.nn as nn
    m = nn.Sequential(
        nn.Linear(600, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    device = "cuda"
    m.to(device)
    x = ch.rand(10, 600)
    y = ch.tensor([1, 0, 1, 0, 1, 0, 1, 1, 0, 1]).unsqueeze(1).float()
    criterion = nn.BCEWithLogitsLoss()
    # Make proper pytorch loader out of (x, y)
    loader = ch.utils.data.DataLoader(ch.utils.data.TensorDataset(x, y), batch_size=3)

    # Compute and get grads for model
    loss = criterion(m(x.to(device)), y.to(device))
    loss.backward()
    flat_grad = []
    for p in m.parameters():
        flat_grad.append(p.grad.detach().view(-1))
    flat_grad = ch.cat(flat_grad)
    m.zero_grad()

    # Get HVP with LiSSA
    hvp = fast_ihvp(m, flat_grad, loader, criterion, device=device)
    print(hvp)

    # Get exact Hessian
    H = compute_hessian(m, loader, criterion, device=device)
    print(H)
