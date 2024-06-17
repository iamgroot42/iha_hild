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
            self.ihvp_module = CGInfluenceModule(
                model=model,
                objective=MyObjective(criterion),
                train_loader=all_train_loader,
                test_loader=None,
                device=self.device,
                damp=damping_eps,
                use_cupy=False
            )
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
        return mi_score


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
