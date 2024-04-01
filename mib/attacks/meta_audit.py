import os

os.environ["KERAS_BACKEND"] = "torch"

import copy
import torch as ch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from tqdm import tqdm
import lightning as L
from torch.utils.data import Dataset
from torchvision import transforms

from mib.attacks.base import Attack
from mib.utils import meta_run_cache
from sklearn.metrics import roc_auc_score

from torch.func import functional_call, vmap, grad

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300


class MetaAudit(Attack):
    """
    Meta-classifier that takes features derived from target model (intermediate activations, etc) for a given point
    and classifies membership.
    """

    def __init__(self, model):
        super().__init__("MetaAudit", model, reference_based=False, whitebox=True)

    def compute_scores(self, x, y, **kwargs) -> np.ndarray:
        meta_clf = kwargs.get("meta_clf", None)
        if meta_clf is None:
            raise ValueError("MetaAudit requires meta_clf to be specified")
        y_hat = meta_clf(x.cuda())
        # Apply sigmoid to get probability
        y_hat = F.sigmoid(y_hat.detach()).cpu()
        return y_hat.numpy()


class MetaModePlain(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 4,
        num_classes_data: int = 10,
        feature_mode: bool = False,
        use_grad_norms: bool = False):
        super().__init__()
        self.feature_mode = feature_mode
        self.use_grad_norms = use_grad_norms

        # For activation
        self.acts_0 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.acts_1 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.acts_2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.acts_3 = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(16),
            nn.Linear(16, hidden_dim),
            nn.ReLU(inplace=True),
        )
        num_acts_use = 4
        self.acts_all = nn.Sequential(
            nn.Linear(num_acts_use * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Attention heads for intermediate activations
        self.attention_acts = nn.MultiheadAttention(hidden_dim, 2)

        # For gradnorms
        if self.use_grad_norms:
            self.gradfc = nn.Sequential(nn.Linear(10, hidden_dim), nn.ReLU(inplace=True))

        # For logits
        self.logits_fc = nn.Sequential(
            nn.Linear(num_classes_data, hidden_dim), nn.ReLU(inplace=True)
        )

        # Embedding layer for data labels
        self.label_embed = nn.Embedding(num_classes_data, hidden_dim)

        # Final connector
        self.latent_dim = hidden_dim * (4 if self.use_grad_norms else 3) + 1
        if not self.feature_mode:
            self.fc = nn.Sequential(
                nn.Linear(self.latent_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, data):
        # For activation
        act_0 = data["activations_0"]
        x_act0 = self.acts_0(act_0)
        act_1 = data["activations_1"]
        x_act1 = self.acts_1(act_1)
        act_2 = data["activations_2"]
        x_act2 = self.acts_2(act_2)
        act_3 = data["activations_3"]
        x_act3 = self.acts_3(act_3)

        # ATTENTION
        # x_acts = ch.stack((x_act0, x_act2, x_act4), 2)
        x_acts = ch.stack((x_act0, x_act1, x_act2, x_act3), 2)
        # 2nd dim is batch size, 1st dim is sequence length
        x_acts = x_acts.permute(2, 0, 1)
        x_acts, _ = self.attention_acts(x_acts, x_acts, x_acts)
        x_acts = x_acts.permute(1, 0, 2)
        # Bring it back into expected shape
        x_acts = x_acts.reshape(x_acts.shape[0], -1)
        # NO ATTENTION
        # x_acts = ch.cat((x_act0, x_act2, x_act4), 1)

        # Activations
        x_acts = self.acts_all(x_acts)

        # For gradnorms
        if self.use_grad_norms:
            gradnorms = data["gradnorms"]
            x_gn = self.gradfc(gradnorms)

        # For logits
        logits = data["logits"]
        x_lg = self.logits_fc(logits)

        # Label embed
        labels = data["labels"]
        x_lb = self.label_embed(labels)

        # Combine them all
        loss = data["loss"]

        if self.use_grad_norms:
            x = ch.cat((x_acts, x_gn, x_lg, x_lb, loss), 1)
        else:
            x = ch.cat((x_acts, x_lg, x_lb, loss), 1)
        if self.feature_mode:
            return x

        x = self.fc(x)

        return x


class MetaModelCNN(nn.Module):
    def __init__(self, hidden_dim: int = 4, num_classes_data: int = 10, feature_mode: bool = False, use_grad_norms: bool = False):
        super().__init__()
        self.feature_mode = feature_mode
        self.use_grad_norms = use_grad_norms
        # For activation
        self.acts_0 = nn.Sequential(
            nn.Conv2d(16, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 4, 5),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(100, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.acts_1 = nn.Sequential(
            nn.Conv2d(32, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 4, 5),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(100, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.acts_2 = nn.Sequential(
            nn.Conv2d(64, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.acts_3 = nn.Sequential(
            nn.Conv2d(128, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(24, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.acts_4 = nn.Sequential(
            nn.Conv2d(128, 6, 5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(24, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.acts_5 = nn.Sequential(nn.Linear(128, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True))
        num_acts_use = 6
        self.acts_all = nn.Sequential(nn.Linear(num_acts_use * hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(inplace=True))

        # Attention heads for intermediate activations
        self.attention_acts = nn.MultiheadAttention(hidden_dim, 2)

        # For gradnorms
        if self.use_grad_norms:
            self.gradfc = nn.Sequential(nn.Linear(108, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.Linear(32, hidden_dim), nn.ReLU(inplace=True))

        # For logits
        self.logits_fc = nn.Sequential(nn.Linear(num_classes_data, hidden_dim), nn.ReLU(inplace=True))

        # Embedding layer for data labels
        self.label_embed = nn.Embedding(num_classes_data, hidden_dim)

        # Final connector
        self.latent_dim = hidden_dim * (4 if self.use_grad_norms else 3) + 1
        if not self.feature_mode:
            self.fc = nn.Sequential(
                nn.Linear(self.latent_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, data):
        # For activation
        act_0 = data["activations_0"]
        x_act0 = self.acts_0(act_0)
        act_1 = data["activations_1"]
        x_act1 = self.acts_1(act_1)
        act_2 = data["activations_2"]
        x_act2 = self.acts_2(act_2)
        act_3 = data["activations_3"]
        x_act3 = self.acts_3(act_3)
        act_4 = data["activations_4"]
        x_act4 = self.acts_4(act_4)
        act_5 = data["activations_5"]
        x_act5 = self.acts_5(act_5)

        # ATTENTION
        # x_acts = ch.stack((x_act0, x_act2, x_act4), 2)
        x_acts = ch.stack((x_act0, x_act1, x_act2, x_act3, x_act4, x_act5), 2)
        # 2nd dim is batch size, 1st dim is sequence length
        x_acts = x_acts.permute(2, 0, 1)
        x_acts, _ = self.attention_acts(x_acts, x_acts, x_acts)
        x_acts = x_acts.permute(1, 0, 2)
        # Bring it back into expected shape
        x_acts = x_acts.reshape(x_acts.shape[0], -1)
        # NO ATTENTION
        # x_acts = ch.cat((x_act0, x_act2, x_act4), 1)

        # Activations
        x_acts = self.acts_all(x_acts)

        # For gradnorms
        if self.use_grad_norms:
            gradnorms = data["gradnorms"]
            x_gn = self.gradfc(gradnorms)

        # For logits
        logits = data["logits"]
        x_lg = self.logits_fc(logits)

        # Label embed
        labels = data["labels"]
        x_lb = self.label_embed(labels)

        # Combine them all
        loss = data["loss"]


        if self.use_grad_norms:
            x = ch.cat((x_acts, x_gn, x_lg, x_lb, loss), 1)
        else:
            x = ch.cat((x_acts, x_lg, x_lb, loss), 1)

        if self.feature_mode:
            return x
        x = self.fc(x)

        return x


class MetaModelSiamese(nn.Module):
    def __init__(self, feature_model):
        super().__init__()
        self.feature_model = feature_model
        latent_dim = self.feature_model.latent_dim
        self.fc = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.ReLU(inplace=True), nn.Linear(latent_dim, 1))

    def get_feature_emb(self, x):
        return self.feature_model(x)

    def forward(self, data_pair, precomputed: bool = False):
        data_left = data_pair["left"]
        data_right = data_pair["right"]
        if precomputed:
            x1 = data_left
            x2 = data_right
        else:
            x1 = self.get_feature_emb(data_left)
            x2 = self.get_feature_emb(data_right)
        x = ch.cat((x1, x2), 1)
        x = self.fc(x)
        return x


def train_meta_classifier(model, num_epochs: int,
                          train_loader, val_loader,
                          lr: float=1e-3,
                          weight_decay: float=5e-4,
                          verbose: bool = True,
                          get_best_val: bool = False,
                          device: str = "cuda",
                          pairwise: bool = False):
    # optimizer = ch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = ch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=0, momentum=0.9)
    # optimizer = ch.optim.SGD(model.parameters(), lr=1e-1, weight_decay=weight_decay, momentum=0.9)
    scheduler = None
    scheduler = ch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    # scheduler = ch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    model.to(device)

    def get_loss_and_acc(y_hat, y):
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = ch.sum((y_hat > 0) == y) / (len(y) * 1.0)
        return loss, acc

    iterator = range(num_epochs)
    if verbose:
        iterator = tqdm(iterator, total=num_epochs)

    # Pick model according to best val AUC
    best_model, best_loss = None, float("inf")

    # Also plot train/val AUC
    train_aucs, val_aucs = [], []

    for epoch in iterator:
        tloss = 0
        preds_train, preds_val = [], []
        labels_train, labels_val = [], []
        model.train()
        nseen = 0
        for batch in train_loader:
            x, y = batch
            y = y.to(device)
            if pairwise:
                x_ = {"left": {}, "right": {}}
                x_["left"] = {k: v.to(device) for k, v in x["left"].items()}
                x_["right"] = {k: v.to(device) for k, v in x["right"].items()}
            else:
                x_ = {k: v.to(device) for k, v in x.items()}

            optimizer.zero_grad()
            y_hat = model(x_).view(-1)
            train_loss, _ = get_loss_and_acc(y_hat, y)
            train_loss.backward()
            optimizer.step()

            with ch.no_grad():
                tloss += train_loss.item()
                preds_train.append(F.sigmoid(y_hat.detach()))
                labels_train.append(y)
                nseen += 1

        # Compute train AUC
        preds_train = ch.cat(preds_train, 0).cpu().numpy()
        labels_train = ch.cat(labels_train, 0).cpu().numpy()
        train_auc = roc_auc_score(labels_train, preds_train)
        train_aucs.append(train_auc)

        if scheduler:
            scheduler.step()

        if val_loader is not None:
            with ch.no_grad():
                model.eval()
                vloss = 0
                for batch in val_loader:
                    x, y = batch
                    y = y.to(device)
                    if pairwise:
                        x_ = {"left": {}, "right": {}}
                        x_["left"] = {k: v.to(device) for k, v in x["left"].items()}
                        x_["right"] = {k: v.to(device) for k, v in x["right"].items()}
                    else:
                        x_ = {k: v.to(device) for k, v in x.items()}

                    y_hat = model(x_).view(-1)
                    val_loss, _ = get_loss_and_acc(y_hat, y)
                    vloss += val_loss.item()
                    preds_val.append(F.sigmoid(y_hat.detach()))
                    labels_val.append(y)
                vloss /= len(val_loader)

            # Compute val AUC
            preds_val = ch.cat(preds_val, 0).cpu().numpy()
            labels_val = ch.cat(labels_val, 0).cpu().numpy()
            val_auc = roc_auc_score(labels_val, preds_val)
            val_aucs.append(val_auc)

            if get_best_val and vloss < best_loss:
                best_loss = vloss
                best_model = copy.deepcopy(model).cpu()

        if verbose:
            if val_loader:
                iterator.set_description(
                    f"Epoch {epoch+1}/{num_epochs} | Loss: {tloss/nseen:.4f} | AUC: {train_auc:.4f} | Validation Loss: {vloss:.4f} | Validation AUC: {val_auc:.4f}"
                )
            else:
                iterator.set_description(
                    f"Epoch {epoch+1}/{num_epochs} | Loss: {tloss/nseen:.4f} | AUC: {train_auc:.4f}"
                )

    # Plot AUCS
    plt.plot(train_aucs, label="Train")
    if val_loader:
        plt.plot(val_aucs, label="Validation")
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.savefig("meta_clf_auc.png")

    # Get best-val-AUC model if requested
    if get_best_val:
        model = best_model.to(device)

    model.eval()
    return model


class PairwiseFeaturesDataset(ch.utils.data.Dataset):
    """
    Wrapper that takes in FeaturesDataset and returns pairs of data points
    """
    def __init__(self, features_dataset, n_task: int):
        self.features_dataset = features_dataset
        self.n_task = n_task

    def __len__(self):
        return self.n_task

    def __getitem__(self, idx):
        idx = np.random.choice(len(self.features_dataset))
        # Make sure idx is within limits
        # idx = idx % len(self.features_dataset)
        features_1, y_1 = self.features_dataset[idx]
        # Get data for another random index
        other_idx = np.random.choice(len(self.features_dataset))
        features_2, y_2 = self.features_dataset[other_idx]
        # Construct pair-wise combinations of this data (features_1, features_2 are each dicts with B-length elements inside)
        feature_branches = {
            "left": {k: [] for k in features_1.keys()},
            "right": {k: [] for k in features_1.keys()},
        }
        matching_labels = []
        for i in range(len(y_1)):
            for j in range(len(y_2)):
                for k, v in features_1.items():
                    feature_branches["left"][k].append(v[i])
                    feature_branches["right"][k].append(features_2[k][j])
                matching_labels.append(1 * (y_1[i] == y_2[j]).item())

        for k, v in feature_branches["left"].items():
            feature_branches["left"][k] = ch.stack(v, 0)
            feature_branches["right"][k] = ch.stack(feature_branches["right"][k], 0)

        return feature_branches, ch.tensor(matching_labels).float()


class FeaturesDataset(ch.utils.data.Dataset):
    """
    Custom pytorch dataclass where (x_data, y_data, y) is stored
    Real feature generation happens in collation step to maximize parallelism in model calls
    """
    def __init__(self, model, x_data, y_data, y_member,
                 batch_size: int, augment: bool = False,
                 cache: bool = False,
                 use_grad_norms: bool = False):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.model.cuda()
        self.x_data = x_data
        self.y_data = y_data
        self.y_member = y_member
        self.batch_size = batch_size
        self.augment = augment
        self.cache = cache
        self.use_grad_norms = use_grad_norms
        if self.cache:
            self.cache_indicator = np.zeros(dtype=bool, shape=len(self))

    def __len__(self):
        return math.ceil(len(self.y_member) / self.batch_size)

    def get_transform(self):
        return transforms.Compose(
                    [
                        # Bring back to [0, 1]
                        transforms.Normalize((-1), (2.0)),
                        # Make image
                        transforms.ToPILImage(),
                        # Apply standard transforms
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5)),
                    ]
                )

    def __getitem__(self, idx):
        if self.cache and not self.augment and self.cache_indicator[idx]:
            # Load from cache
            path = os.path.join(meta_run_cache(), f"{idx}.pt")
            data = ch.load(path)
            return data["features"], data["y"]
        else:
            idx_start = idx * self.batch_size
            idx_end = idx_start + self.batch_size

            with ch.no_grad():
                x_ = self.x_data[idx_start: idx_end]
                y_ = self.y_data[idx_start: idx_end]

                # Apply transforms if requested
                if self.augment:
                    tf = self.get_transform()
                    # Apply instance-wise transform
                    x_ = ch.stack([tf(z) for z in x_], 0)

            features = self.get_signals(x_, y_)

            # If not augmentation-based, dump to cache
            if self.cache and not self.augment:
                # Dump file to cache
                path = os.path.join(meta_run_cache(), f"{idx}.pt")
                ch.save({"features": features, "y": self.y_member[idx_start: idx_end]}, path)
                self.cache_indicator[idx] = True

        return features, self.y_member[idx_start: idx_end]

    def get_signals(self, x, y):
        x_, y_ = x.cuda(), y.cuda()
        # Get model activations
        with ch.no_grad():
            acts = self.model(x_, get_all=True)
        # Also get loss
        self.model.zero_grad()
        y_hat = self.model(x_)

        if self.use_grad_norms:
            # Faster version with vmap

            self.model.zero_grad()
            params = {k: v.detach() for k, v in self.model.named_parameters()}
            buffers = {k: v.detach() for k, v in self.model.named_buffers()}

            def compute_loss(params, buffers, sample, target):
                batch = sample.unsqueeze(0).cuda()
                targets = target.unsqueeze(0).cuda()
                predictions = functional_call(self.model, (params, buffers), (batch,))
                loss = ch.nn.CrossEntropyLoss()(predictions, targets)
                return loss

            ft_compute_grad = grad(compute_loss)
            ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, x, y)

            gradnorms = []
            for k, v in ft_per_sample_grads.items():
                v_flat = v.detach().reshape(v.shape[0], -1)
                gradnorms.append(ch.norm(v_flat, 2, dim=1))
            gradnorms = ch.stack(gradnorms, 1).cpu()
            self.model.zero_grad()
            losses = ch.nn.CrossEntropyLoss(reduction="none")(y_hat, y_).detach().cpu().unsqueeze(1)

            """
            # Might as well take note of logits
            losses, gradnorms = [], []
            # Compute element-wise loss and grad norm
            for i in range(len(y_hat)):
                self.model.zero_grad()
                loss = ch.nn.CrossEntropyLoss()(y_hat[i].unsqueeze(0), y_[i].unsqueeze(0))
                loss.backward(retain_graph=True)
                losses.append(loss.detach())
                gnorms = []
                for p in self.model.parameters():
                    if p.grad is not None:
                        gnorms.append(ch.norm(p.grad.data.detach(), 2))
                gnorms = ch.stack(gnorms).cpu()
                gradnorms.append(gnorms)
            losses = ch.stack(losses).cpu().unsqueeze(1)
            gradnorms = ch.stack(gradnorms).cpu()
            """
        else:
            losses = ch.nn.CrossEntropyLoss(reduction="none")(y_hat, y_).detach().cpu().unsqueeze(1)

        m = {
            "logits": y_hat.detach().cpu(),
            "loss": losses,
            "labels": y_.cpu(),
        }
        if self.use_grad_norms:
            m["gradnorms"] = gradnorms
        for i in range(len(acts)):
            m[f"activations_{i}"] = acts[i].detach().cpu()

        return m


def dict_collate_fn(batch):
    y = ch.cat([b[1] for b in batch])
    # List of dicts. Convert to dict of tensors
    x = {k: [] for k in batch[0][0].keys()}
    for b in batch:
        for k, v in b[0].items():
            x[k].append(v)
    for k, v in x.items():
        x[k] = ch.cat(v, 0)

    return x, y


def dict_collate_fn_pairwise(batch):
    y = ch.cat([b[1] for b in batch])
    # List of dicts. Convert to dict of tensors
    x = {}
    x["left"] = {k: [] for k in batch[0][0]["left"].keys()}
    x["right"] = {k: [] for k in batch[0][0]["right"].keys()}
    for b in batch:
        for k in b[0]["left"].keys():
            x["left"][k].append(b[0]["left"][k])
            x["right"][k].append(b[0]["right"][k])
    for k, v in x["left"].items():
        x["left"][k] = ch.cat(v, 0)
        x["right"][k] = ch.cat(x["right"][k], 0)

    return x, y


def train_meta_clf(meta_clf, model, x_both, y,
                   batch_size: int = 32,
                   num_epochs: int = 10,
                   val_points: int = 1000,
                   device: str = "cuda",
                   augment: bool = False,
                   pairwise: bool = False,
                   use_grad_norms: bool = False):
    x, y_data = x_both

    # Sample points for validation using train-test split
    val_idx = np.random.choice(len(x), val_points, replace=False)
    train_idx = np.array([i for i in range(len(x)) if i not in val_idx])

    x_train, y_data_train, y_train = x[train_idx], y_data[train_idx], y[train_idx]
    x_val, y_data_val, y_val = x[val_idx], y_data[val_idx], y[val_idx]

    """
    x_train = [x for sublist in x_train for x in sublist]
    x_val = [x for sublist in x_val for x in sublist]
    n_reps = len(x_train[0])
    # y iy not list of lists, but same label holds for each internal list. So [0, 1, 2] shoulbd become [0, 0, 0, 1, 1, 1, 2, 2, 2] if internal list had 3 elements
    y_train = np.repeat(y_train, n_reps)
    y_val = np.repeat(y_val, n_reps)
    """
    # Cache is run-specific. Clear its contents for now
    for f in os.listdir(meta_run_cache()):
        os.remove(os.path.join(meta_run_cache(), f))

    if pairwise:
        collate_fn_use = dict_collate_fn_pairwise
    else:
        collate_fn_use = dict_collate_fn

    # Make loader out of (x, y) data
    ds_train = FeaturesDataset(
        model,
        x_train,
        y_data_train,
        ch.from_numpy(y_train).float(),
        batch_size=batch_size,
        augment=augment,
        use_grad_norms=use_grad_norms
    )
    if pairwise:
        ds_train = PairwiseFeaturesDataset(ds_train, n_task=20_000)
    train_loader = ch.utils.data.DataLoader(ds_train, batch_size=4, shuffle=True,
                                            num_workers=4,
                                            prefetch_factor=4,
                                            pin_memory=True,
                                            collate_fn=collate_fn_use)

    # Make loader out of (x, y) data
    ds_val = FeaturesDataset(
        model, x_val, y_data_val,
        ch.from_numpy(y_val).float(),
        batch_size=batch_size,
        use_grad_norms=use_grad_norms)
    if pairwise:
        ds_val = PairwiseFeaturesDataset(ds_val, n_task=3_000)
    val_loader = ch.utils.data.DataLoader(ds_val, batch_size=4, shuffle=False,
                                          num_workers=4,
                                          prefetch_factor=4,
                                          pin_memory=True,
                                          collate_fn=collate_fn_use)

    meta_clf_trained = train_meta_classifier(meta_clf, num_epochs, train_loader, val_loader,
                                             # get_best_val=True,
                                             get_best_val=False,
                                             device=device,
                                             pairwise=pairwise)

    """
    wrapped_meta = LitMetaModel(meta_clf)
    trainer = L.Trainer(max_epochs=num_epochs, accelerator="cuda", log_every_n_steps=10)
    trainer.fit(model=wrapped_meta, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Extract trained model from wrapped model
    meta_clf_trained = wrapped_meta.model
    """
    return meta_clf_trained


def test_meta(meta_clf, loader, gt_labels, args, META_DEVICE):

    test_preds = []
    collected_features = []
    for batch in loader:
        x, y = batch[0], batch[1]
        x_ = {k: v.to(META_DEVICE) for k, v in x.items()}

        if args.pairwise:
            embed = meta_clf.get_feature_emb(x_).detach()
            collected_features.append(embed)
        else:
            preds = ch.nn.functional.sigmoid(meta_clf(x_)).detach().cpu().numpy()
            # preds = preds.mean(0)
            test_preds.extend(preds)

    if args.pairwise:
        collected_features = ch.cat(collected_features, 0)
        for i in range(len(collected_features)):
            # Construct inputs (collected_features[i], collected_features[j]) for varying j
            inputs = {"left": collected_features[i].unsqueeze(0).repeat(len(collected_features), 1), "right": collected_features}
            outputs = ch.nn.functional.sigmoid(meta_clf(inputs, precomputed=True)).detach().cpu().numpy()
            # Aggregate prediction as weighted sum of predicted output (which tells you whether same class or not) and corresponding ground-truth labels

            # We essentially want P(y=1)
            weighted_pred = np.zeros_like(outputs)
            weighted_pred[gt_labels == 1] = outputs[gt_labels == 1]
            weighted_pred[gt_labels == 0] = 1 - outputs[gt_labels == 0]
            # Of course, skip ith point (like, do not count it in mean)
            weighted_pred[i] = None
            test_preds.append(np.nanmean(weighted_pred))

    return test_preds
