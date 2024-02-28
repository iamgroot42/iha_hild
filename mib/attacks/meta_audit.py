import os

os.environ["KERAS_BACKEND"] = "torch"


import torch as ch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import lightning as L

from mib.attacks.base import Attack


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
    def __init__(self, input_dim, hidden_dims = [16, 8]):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MetaModelCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 8, 3)
        output_dim = (input_dim - 3) // 2 + 1
        # output_dim = output_dim * output_dim * hidden_dim (TODO - make dynamic later)
        output_dim = 1800
        self.fc1 = nn.Linear(output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + 1, 1)

    def forward(self, x, loss):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, (2, 2))
        # Flatten
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = ch.cat((x, loss.unsqueeze(1)), 1)
        x = self.fc2(x)
        return x


class LitMetaModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _get_loss_and_acc(self, y_hat, y):
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        acc = ch.sum((y_hat > 0) == y) / (len(y) * 1.0)
        return loss, acc

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, loss, y = batch
        # x = x.view(x.size(0), -1)
        y_hat = self.model(x, loss).view(-1)
        loss, acc = self._get_loss_and_acc(y_hat, y)

        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, loss, y = batch
        y_hat = self.model(x, loss).view(-1)
        loss, acc = self._get_loss_and_acc(y_hat, y)

        self.log("val_acc", acc, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = ch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train_meta_clf(meta_clf, x, y, batch_size: int = 32, num_epochs: int = 10, val_points: int = 1000):
    # Sample points for validation using train-test split
    val_idx = np.random.choice(len(x[0]), val_points, replace=False)
    train_idx = np.array([i for i in range(len(x[0])) if i not in val_idx])
    
    x_train = [ch.from_numpy(x[i][train_idx]) for i in range(len(x))]
    x_val = [ch.from_numpy(x[i][val_idx]) for i in range(len(x))]
    y_train, y_val = y[train_idx], y[val_idx]

    # Make loader out of (x, y) data
    ds_train = ch.utils.data.TensorDataset(*x_train, ch.from_numpy(y_train).float())
    train_loader = ch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)

    # Make loader out of (x, y) data
    ds_val = ch.utils.data.TensorDataset(*x_val, ch.from_numpy(y_val).float())
    val_loader = ch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=2)

    wrapped_meta = LitMetaModel(meta_clf)
    trainer = L.Trainer(max_epochs=num_epochs, accelerator="cuda", log_every_n_steps=10)
    trainer.fit(model=wrapped_meta, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # Extract trained model from wrapped model
    meta_clf_trained = wrapped_meta.model
    return meta_clf_trained
