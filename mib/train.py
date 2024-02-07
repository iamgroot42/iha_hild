import numpy as np
import os
from torch import nn
import torchvision
import numpy as np
from tqdm import tqdm
import torch as ch
import argparse
import copy
import torchvision
from torchvision import transforms

from mib.models.utils import get_model


def get_data(
    seed: int = 2024,
    pkeep: float = 0.5,
    num_experiments: int = 8,
    exp_id: int = None,
    just_want_data: bool = False,
):
    """
    For random split generation, follow same setup as Carlini/Shokri.
    This is the function to generate subsets of the data for training models.

    First, we get the training dataset.

    Then, we compute the subset. This works in one of two ways.

    1. If we have a seed, then we just randomly choose examples based on
       a prng with that seed, keeping pkeep fraction of the data.

    2. Otherwise, if we have an experiment ID, then we do something fancier.
       If we run each experiment independently then even after a lot of trials
       there will still probably be some examples that were always included
       or always excluded. So instead, with experiment IDs, we guarantee that
       after num_experiments are done, each example is seen exactly half
       of the time in train, and half of the time not in train.

    """
    transforms_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )
    transforms_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    )

    train_data = torchvision.datasets.CIFAR10(
        root="/u/as9rw/work/auditing_mi",
        train=True,
        download=True,
        transform=transforms_train,
    )
    test_data = torchvision.datasets.CIFAR10(
        root="/u/as9rw/work/auditing_mi",
        train=False,
        download=True,
        transform=transforms_test,
    )

    if just_want_data:
        return train_data, test_data

    def get_keep(data, sd: int):
        if num_experiments is not None:
            np.random.seed(sd)
            keep = np.random.uniform(0, 1, size=(num_experiments, len(data.data)))
            order = keep.argsort(0)
            keep = order < int(pkeep * num_experiments)
            keep = np.array(keep[exp_id], dtype=bool)
        else:
            np.random.seed(sd)
            keep = np.random.uniform(0, 1, size=len(data.data)) <= pkeep
        # Create indices corresponding to keep
        indices = np.arange(len(data.data))
        return indices[keep]

    # Split such that every datapoint seen in half of experiments
    train_keep = get_keep(train_data, seed)
    test_keep = get_keep(test_data, seed + 1)

    return train_keep, test_keep, train_data, test_data


"""
def load_data(num_train_points: int, num_test_points: int):
    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    all_data = torchvision.datasets.CIFAR10(
        root=".", train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=".", train=False, download=True, transform=transform
    )
    all_features = np.concatenate([all_data.data, test_data.data], axis=0)
    all_targets = np.concatenate([all_data.targets, test_data.targets], axis=0)

    all_data.data = all_features
    all_data.targets = all_targets

    if num_train_points is None or num_test_points is None:
        return all_data

    all_index = np.arange(len(all_data))
    train_index = np.random.choice(all_index, num_train_points, replace=False)
    test_index = np.random.choice(
        [i for i in all_index if i not in train_index], num_test_points, replace=False
    )
    return all_data, train_index, test_index
"""


def get_loader(dataset, indices, batch_size: int, start_seed: int = 42, shuffle: bool = True):
    num_workers = 2
    loader = ch.utils.data.DataLoader(
        ch.utils.data.Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=lambda worker_id: np.random.seed(start_seed + worker_id),
    )
    return loader


def get_loaders(
    all_data, train_index, test_index, batch_size: int, start_seed: int = 2024
):
    train_loader = get_loader(all_data, train_index, batch_size, start_seed)
    test_loader = get_loader(all_data, test_index, batch_size, start_seed)
    return train_loader, test_loader


def train_model(
    model,
    criterion,
    train_loader,
    test_loader,
    learning_rate: float,
    epochs: int,
    device="cuda",
    verbose: bool = True,
    weight_decay: float = 5e-4,
    loss_multiplier: float = 1.0,
    best_n: int = 1
):
    model.train()

    # Set the loss function and optimizer
    # optimizer = ch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = ch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9
    )
    scheduler = ch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loop over each epoch
    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator)

    model_ckpts, model_losses, model_accs = [], [], []
    for epoch_idx in iterator:
        model.train()
        train_loss = 0
        train_acc = 0
        samples_seen = 0
        # Loop over the training set
        for data, target in train_loader:
            # Move data to the device
            if type(data) == list:
                data_use = []
                for d in data:
                    if type(d) == list:
                        data_use.append([d.to(device, non_blocking=True) for d in d])
                    else:
                        data_use.append(d.to(device, non_blocking=True))
            else:
                data_use = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            samples_seen += len(target)

            # Set the gradients to zero
            optimizer.zero_grad(set_to_none=True)

            # Get the model output
            output = model(data_use)

            # Calculate the loss
            binary_case = False
            if output.shape[1] == 1:
                # BCE loss case
                binary_case = True
                loss = criterion(output.squeeze(), target.float())
            else:
                # CE loss case
                loss = criterion(output, target.long())
            loss *= loss_multiplier

            # Perform the backward pass
            loss.backward()
            # Take a step using optimizer
            optimizer.step()

            # Add the loss to the total loss
            train_loss += loss.item()

            # Computing accuracy
            with ch.no_grad():
                if binary_case:
                    pred = output.data.squeeze() > 0.5
                    train_acc += pred.eq(target.data.view_as(pred)).sum()
                else:
                    pred = output.data.max(1, keepdim=True)[1]
                    train_acc += pred.eq(target.data.view_as(pred)).sum()

        if test_loader is not None:
            test_acc, test_loss = evaluate_model(
                model, test_loader, criterion, device=device
            )
            if len(model_ckpts) < best_n:
                model_ckpts.append(copy.deepcopy(model).cpu())
                model_losses.append(test_loss)
                model_accs.append(test_acc)
            else:
                if test_loss < max(model_losses):
                    # Kick out the model with the highest loss
                    idx = np.argmax(model_losses)
                    model_ckpts[idx] = copy.deepcopy(model).cpu()
                    model_losses[idx] = test_loss
                    model_accs[idx] = test_acc

        if verbose:
            iterator.set_description(
                f"Epoch: {epoch_idx+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Train Accuracy: {100 * train_acc / samples_seen:.2f} | Test Loss: {test_loss:.4f} | Test Accuracy: {100 * test_acc:.2f}"
            )

        # Scheduler step
        scheduler.step()

    if test_loader:
        print("Best test accuracy: ", max(model_accs))

    if best_n == 1:
        return model_ckpts[0], model_accs[0], model_losses[0]
    return model_ckpts, model_accs, model_losses


@ch.no_grad()
def evaluate_model(model, test_loader, criterion, device="cuda"):
    # Validate the performance of the model
    model.eval()
    # Assigning variables for computing loss and accuracy
    loss, acc = 0, 0

    for data, target in test_loader:
        # Moving data and target to the device
        data, target = data.to(device, non_blocking=True), target.to(
            device, non_blocking=True
        )
        # Cast target to long tensor
        target = target.long()

        # Computing output and loss
        output = model(data)
        loss += criterion(output, target).item()

        # Computing accuracy
        pred = output.data.max(1, keepdim=True)[1]
        acc += pred.eq(target.data.view_as(pred)).sum()

    # Averaging the losses
    loss /= len(test_loader)

    # Calculating accuracy
    acc = float(acc) / len(test_loader.dataset)

    return acc, loss


def main(save_dir: str, args):
    n_models = args.num_models
    same_init = args.init_ref
    # Follow setup of LiRA
    # 50% of data is used for training model
    # Other 50% used to train quantile model, and also serve as non-members

    # CIFAR
    pkeep = 0.5

    # Train target model
    dataset = "CIFAR10"
    device = "cuda"

    save_dir_use = save_dir
    if same_init is not None:
        save_dir_use = f"{save_dir_use}/same_init/{same_init}"

    num_trained = 0
    for i in range(n_models):
        # Skip if model already exists
        if os.path.exists(f"{save_dir_use}/{i}.pt"):
            print("Skipping model", i)
            continue

        # Get model
        model, criterion, hparams = get_model(args.model_arch, n_classes=10)
        model.to(device)
        batch_size = hparams["batch_size"]
        learning_rate = hparams["learning_rate"]
        epochs = hparams["epochs"]

        if same_init is not None:
            # Use same initialization as model with index same_init
            model_init = ch.load(f"{save_dir}/{same_init}.pt")["model_init"]
            model.load_state_dict(model_init)
            model.to(device)
        else:
            # Random initialization
            model_init = copy.deepcopy(model.state_dict())

        # Compile model (Faster training)
        model = ch.compile(model)

        # Get data
        train_index, test_index, train_data, test_data = get_data(
            num_experiments=n_models, pkeep=pkeep, exp_id=i
        )

        # Get loaders
        train_loader = get_loader(train_data, train_index, batch_size)
        test_loader = get_loader(test_data, test_index, batch_size)
        # Get loaders
        # train_loader, test_loader = get_loaders(all_data, train_index, test_index, batch_size)

        # Train model
        model, best_acc, best_loss = train_model(
            model,
            criterion,
            train_loader,
            test_loader,
            learning_rate,
            epochs,
            best_n=args.best_n,
        )

        # Make sure folder directory exists
        os.makedirs(save_dir_use, exist_ok=True)

        if args.best_n == 1:
            # Save model dictionary, along with information about train_index and test_index
            ch.save(
                {
                    "model_init": model_init,
                    "model": model._orig_mod.state_dict(),
                    "train_index": train_index,
                    "test_index": test_index,
                    "loss": best_loss,
                    "acc": best_acc,
                },
                f"{save_dir_use}/{i}.pt",
            )
        else:
            # Make sure folder directory exists
            subdir = os.path.join(save_dir_use, f"best_{args.best_n}")
            os.makedirs(subdir, exist_ok=True)

            for j, (m, acc, l) in enumerate(zip(model, best_acc, best_loss)):
                ch.save(
                    {
                        "model_init": model_init,
                        "model": m._orig_mod.state_dict(),
                        "train_index": train_index,
                        "test_index": test_index,
                        "loss": l,
                        "acc": acc,
                    },
                    f"{subdir}/{i}_{j+1}.pt",
                )

        # Break if we have trained enough models
        num_trained += 1
        if num_trained >= args.num_train:
            break


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_arch", type=str, default="wide_resnet_28_2")
    args.add_argument("--num_models", type=int, default=128, help="Total number of models (data splits will be created accordingly)")
    args.add_argument("--num_train", type=int, default=128, help="Number of models to train (out of num_models)")
    args.add_argument("--best_n", type=int, default=1, help="Of all checkpoints, keep the best n.")
    args.add_argument(
        "--init_ref",
        type=int,
        default=None,
        help="If not None, use same initialization as model with this index",
    )
    args = args.parse_args()

    if args.num_train > args.num_models:
        raise ValueError("num_train cannot be greater than num_models")
    if args.best_n < 1:
        raise ValueError("best_n must be >= 1")

    main(f"/u/as9rw/work/auditing_mi/models/{args.model_arch}", args)
