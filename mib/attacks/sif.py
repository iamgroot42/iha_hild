"""
    Self Influence Function (SIF)
"""
import numpy as np
import torch as ch

from mib.attacks.base import Attack
from torch.autograd import grad
from torch_influence import BaseObjective
from torch_influence import (
    LiSSAInfluenceModule,
    CGInfluenceModule,
)
from mib.attacks.theory_new import compute_hessian


class SIF(Attack):
    """
    Direct computation of SIF (g H^{-1] g)
    """

    def __init__(self, model, criterion, **kwargs):
        all_train_loader = kwargs.get("all_train_loader", None)
        approximate = kwargs.get("approximate", False)
        hessian = kwargs.get("hessian", None)
        if all_train_loader is None:
            raise ValueError(
                "SIF requires all_train_loader to be specified"
            )
        super().__init__(
            "SIF",
            model,
            criterion,
            reference_based=False,
            requires_trace=False,
            whitebox=True,
            uses_hessian = not approximate
        )

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
                        return criterion(
                            outputs.squeeze(1), batch[1].float()
                        )  # mean reduction required
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

            """
            self.ihvp_module = LiSSAInfluenceModule(
                model=model,
                objective=MyObjective(),
                train_loader=all_train_loader,
                test_loader=None,
                device="cuda",
                repeat=4,
                depth=100,  # 5000 for MLP and Transformer, 10000 for CNN
                scale=25,
                damp=2e-1,
            )
            """
            self.ihvp_module = CGInfluenceModule(
                model=model,
                objective=MyObjective(),
                train_loader=all_train_loader,
                test_loader=None,
                device="cuda",
                damp=1e-2,
            )
            # """
        else:
            if hessian is None:
                exact_H = compute_hessian(
                    model, all_train_loader, self.criterion, device="cuda"
                )
                self.hessian = exact_H.cpu().clone().detach()
            else:
                self.hessian = hessian

            L, Q = ch.linalg.eigh(self.hessian)
            eps = 1e-1

            qualifying_indices = ch.abs(L) > eps
            Q_select = Q[:, qualifying_indices]
            self.H_inverse = Q_select @ ch.diag(1 / L[qualifying_indices]) @ Q_select.T

    def collect_grad_on_all_data(self, loader):
        cumulative_gradients = None
        for x, y in loader:
            # Zero-out accumulation
            self.model.zero_grad()
            # Compute gradients
            x, y = x.cuda(), y.cuda()
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
        logits = self.model(point_x.cuda())
        if logits.shape[1] == 1:
            loss = self.criterion(logits.squeeze(1), point_y.float().cuda())
        else:
            loss = self.criterion(logits, point_y.cuda())
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

        # Factor out S/(2L*) parts out of both terms as common
        grad, ret_loss = self.get_specific_grad(x, y)

        if self.approximate:
            datapoint_ihvp = self.ihvp_module.inverse_hvp(grad)
        else:
            datapoint_ihvp = (self.H_inverse @ grad.cpu()).cuda()

        score = ch.dot(datapoint_ihvp, grad).cpu().item()
        return score
