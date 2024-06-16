"""
    Train a m-layer MLP with multivariate Gaussian distribution as data.
    Allows to work in simplistic setting and have potentially closed-form solutions for these attacks.
"""

import numpy as np
import torch as ch

from mib.attacks.base import Attack
from torch.autograd import grad
from torch_influence import BaseObjective
from torch_influence import AutogradInfluenceModule, LiSSAInfluenceModule, HVPModule, CGInfluenceModule


class MyObjective(BaseObjective):
    def __init__(self, criterion):
        self._criterion = criterion

    def train_outputs(self, model, batch):
        return model(batch[0])

    def train_loss_on_outputs(self, outputs, batch):
        if outputs.shape[1] == 1:
            return self._criterion(
                outputs.squeeze(1), batch[1].float()
            )  # mean reduction required
        else:
            return self._criterion(outputs, batch[1])  # mean reduction required

    def train_regularization(self, params):
        return 0

    def test_loss(self, model, params, batch):
        output = model(batch[0])
        # no regularization in test loss
        if output.shape[1] == 1:
            return self._criterion(output.squeeze(1), batch[1].float())
        else:
            return self._criterion(output, batch[1])


class ProperTheoryRef(Attack):
    """
    I1 + I3 based term that makes assumption about relationship between I2 and I3.
    """
    def __init__(self, model, criterion, device: str = "cuda", **kwargs):
        all_train_loader = kwargs.get("all_train_loader", None)
        approximate = kwargs.get("approximate", False) # Use approximate iHVP instead of exact?
        hessian = kwargs.get("hessian", None) # Precomputed Hessian, if available
        damping_eps = kwargs.get("damping_eps", 2e-1) # Damping (or cutoff for low-rank approximation)
        low_rank = kwargs.get("low_rank", False) # Use low-rank approximation for Hessian?
        save_compute_trick = kwargs.get("save_compute_trick", False) # Use L1 = L0 + 1/n l(z) trick to reduce iHVP calls per point from 2 to 1

        if all_train_loader is None:
            raise ValueError("ProperTheoryRef requires all_train_loader to be specified")
        super().__init__(
            "ProperTheoryRef",
            model,
            criterion,
            device=device,
            reference_based=False,
            requires_trace=False,
            whitebox=True,
            uses_hessian=not approximate,
        )

        self.approximate = approximate
        self.model.to(self.device)

        self.all_data_grad = self.collect_grad_on_all_data(all_train_loader)

        # Exact Hessian
        if self.approximate:
            # LiSSA - 40s/iteration
            # GC - 38s/iteration
            """
            self.ihvp_module = LiSSAInfluenceModule(
                model=model,
                objective=MyObjective(criterion),
                train_loader=all_train_loader,
                test_loader=None,
                device=self.device,
                repeat=4,
                depth=100,  # 5000 for MLP and Transformer, 10000 for CNN
                scale=25,
                damp=damping_eps,
            )
            """
            self.ihvp_module = CGInfluenceModule(
                model=model,
                objective=MyObjective(criterion),
                train_loader=all_train_loader,
                test_loader=None,
                device=self.device,
                damp=damping_eps,
            )
            # """
        else:
            if hessian is None:
                exact_H = compute_hessian(model, all_train_loader, self.criterion, device=self.device)
                self.hessian = exact_H.cpu().clone().detach()
            else:
                self.hessian = hessian

            L, Q = ch.linalg.eigh(self.hessian)

            if low_rank:
                # Low-rank approximation
                qualifying_indices = ch.abs(L) > damping_eps
                Q_select = Q[:, qualifying_indices]
                self.H_inverse = Q_select @ ch.diag(1 / L[qualifying_indices]) @ Q_select.T
            else:
                # Damping
                L += damping_eps
                self.H_inverse = Q @ ch.diag(1 / L) @ Q.T

        self.l1_ihvp = None
        if save_compute_trick:
            # n * L1 = (n-1) * L0 + l(z)
            # L0 = (n * L1 - l(z) / (n-1)
            if self.approximate:
                self.l1_ihvp = self.ihvp_module.inverse_hvp(self.all_data_grad)
            else:
                self.l1_ihvp = (self.H_inverse @ self.all_data_grad.cpu()).to(self.device)

    def collect_grad_on_all_data(self, loader):
        cumulative_gradients = None
        for x, y in loader:
            # Zero-out accumulation
            self.model.zero_grad()
            # Compute gradients
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            if logits.shape[1] == 1:
                loss = self.criterion(logits.squeeze(), y.float()) * len(x)
            else:
                loss = self.criterion(logits, y) * len(x)
            loss.backward()
            flat_grad = []
            for p in self.model.parameters():
                flat_grad.append(p.grad.detach().view(-1))
            # Flatten out gradients
            flat_grad = ch.cat(flat_grad)
            # Accumulate in higher precision
            if cumulative_gradients is None:
                cumulative_gradients = ch.zeros_like(flat_grad)
            cumulative_gradients += flat_grad
        self.model.zero_grad()
        cumulative_gradients /= len(loader.dataset)
        return cumulative_gradients

    def get_specific_grad(self, point_x, point_y):
        self.model.zero_grad()
        logits = self.model(point_x.to(self.device))
        if logits.shape[1] == 1:
            loss = self.criterion(logits.squeeze(1), point_y.float().to(self.device))
        else:
            loss = self.criterion(logits, point_y.to(self.device))
        ret_loss = loss.item()
        loss.backward()
        flat_grad = []
        for p in self.model.parameters():
            flat_grad.append(p.grad.detach().view(-1))
        flat_grad = ch.cat(flat_grad)
        self.model.zero_grad()
        return flat_grad, ret_loss

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        x, y = x.to(self.device), y.to(self.device)
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
            # Trick to skip computing L0 for all records. Compute L1 (across all data), and then directly calculate L0 using current grad
            # We are passing 'is_train' flag here but this is not cheating- could be replaced with repeated L0 computation, but would be unnecessarily expensive
            # all_other_data_grad = self.all_data_grad - (grad / num_samples)
            all_other_data_grad = (self.all_data_grad * num_samples - grad) / (num_samples - 1)
        else:
            all_other_data_grad = self.all_data_grad

        if self.approximate:
            # H-1 * grad(l(z))
            datapoint_ihvp = self.ihvp_module.inverse_hvp(grad)
            # H-1 * grad(L0(z))
            if self.l1_ihvp is not None:
                ihvp_alldata = (num_samples * self.l1_ihvp - datapoint_ihvp) / (num_samples - 1)
            else:
                ihvp_alldata   = self.ihvp_module.inverse_hvp(all_other_data_grad)
        else:
            # H-1 * grad(l(z))
            datapoint_ihvp = (self.H_inverse @ grad.cpu()).to(self.device)
            # H-1 * grad(L0(z))
            if self.l1_ihvp is not None:
                ihvp_alldata = (num_samples * self.l1_ihvp - datapoint_ihvp) / (num_samples - 1)
            else:
                ihvp_alldata   = (self.H_inverse @ all_other_data_grad.cpu()).to(self.device)

        I2 = ch.dot(datapoint_ihvp, datapoint_ihvp).cpu().item() / num_samples
        I3 = ch.dot(ihvp_alldata, datapoint_ihvp).cpu().item() * 2

        I2 /= learning_rate
        I3 /= learning_rate

        mi_score = I1 - (I2 + I3)
        # return -I3
        return mi_score


def fast_ihvp(model, vec, loader, criterion, device: str = "cpu"):
    """
        Use LiSSA to compute HVP for a given model and dataloader
    """
    module = LiSSAInfluenceModule(
        model=model,
        objective=MyObjective(criterion),
        train_loader=loader,
        test_loader=None,
        device=device,
        damp=0,
        repeat=20,
        depth=100,  # 5000 for MLP and Transformer, 10000 for CNN
        scale=50,  # test in {10, 25, 50, 100, 150, 200, 250, 300, 400, 500} for convergence
    )

    # Get projection of vec onto inverse-Hessian
    ihvp = module.inverse_hvp(vec)

    """
    hvp_module = HVPModule(model, MyObjective(criterion), loader, device=self.device)
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
    model.zero_grad()

    module = AutogradInfluenceModule(
        model=model,
        objective=MyObjective(criterion),
        train_loader=loader,
        test_loader=None,
        device=device,
        damp=0,
        store_as_hessian=True,
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
