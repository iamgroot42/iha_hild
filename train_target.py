import numpy as np
import os
from torch import nn
import torchvision
import numpy as np
from tqdm import tqdm
import torch as ch
import torchvision
from torchvision import transforms

from wide_resnet import Wide_ResNet


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


def get_model_and_criterion(dataset_name: str, device: str = "cuda"):
    m = Wide_ResNet(28, 2, 10)
    m.to(device)
    return m, nn.CrossEntropyLoss()


def get_loaders(all_data, train_index, test_index, batch_size: int, start_seed: int = 2023):
    num_workers = 4
    train_loader = ch.utils.data.DataLoader(
        ch.utils.data.Subset(all_data, train_index),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=16,
        worker_init_fn = lambda worker_id: np.random.seed(start_seed + worker_id)
    )
    test_loader = ch.utils.data.DataLoader(
        ch.utils.data.Subset(all_data, test_index),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=16,
        worker_init_fn = lambda worker_id: np.random.seed(start_seed + worker_id)
    )
    return train_loader, test_loader


def train_model(
    model,
    criterion,
    train_loader,
    test_loader,
    learning_rate: float,
    epochs: int,
    device="cuda",
):
    model.train()

    # Set the loss function and optimizer
    optimizer = ch.optim.Adam(model.parameters(), lr=learning_rate)
    # Loop over each epoch
    iterator = tqdm(range(epochs))
    for epoch_idx in iterator:
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

        iterator.set_description(
            f"Epoch: {epoch_idx+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Accuracy: {100 * train_acc / samples_seen:.2f}"
        )

        if epoch_idx % 5 == 0:
            print(f"Epoch: {epoch_idx+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Accuracy: {100 * train_acc / samples_seen:.2f}")

    # Validate the performance of the model
    model.eval()
    # Assigning variables for computing loss and accuracy
    loss, acc = 0, 0

    # If not test-loader, shift to CPU and return directly
    if test_loader is None:
        model.to("cpu")
        return model

    # Disable gradient calculation to save memory
    with ch.no_grad():
        for data, target in test_loader:
            # Moving data and target to the device
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
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
        print("Test Loss: {:.4f}, Test Accuracy: {:.2f}%".format(loss, 100.0 * acc))

    # Move the model back to the CPU to save memory
    model.to("cpu")

    return model


def main(save_dir: str, n_models: int = 1):
    # Follow setup of LiRA
    # 50% of data is used for training model
    # Other 50% used to train quantile model, and also serve as non-members

    # CIFAR
    num_train_points = 25000
    num_test_points = 5000
    batch_size = 64
    learning_rate = 0.001
    epochs = 20

    # Train target model
    dataset = "CIFAR10"
    device = "cuda"

    for i in range(1, n_models + 1):
        # Get model
        model, criterion = get_model_and_criterion(dataset, device=device)
        model = ch.compile(model)

        # Get data
        all_data, train_index, test_index = load_data(num_train_points, num_test_points)

        # Get loaders
        train_loader, test_loader = get_loaders(
            all_data, train_index, test_index, batch_size
        )

        # Train model
        model = train_model(
            model, criterion, train_loader, test_loader, learning_rate, epochs
        )

        # Make sure folder directory exists
        os.makedirs(save_dir, exist_ok=True)
        # Save model dictionary, along with information about train_index and test_index
        ch.save(
            {
                "model": model._orig_mod.state_dict(),
                "train_index": train_index,
                "test_index": test_index,
            },
            f"{save_dir}/{i}.pt",
        )


if __name__ == "__main__":
    main("target_model", 1)
