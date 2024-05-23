import torch as ch
import numpy as np
from mib.utils import get_data_source
import os

from mib.dataset.base import Dataset


def load_purchase100_file(name: str, num_train: int):
    # TODO - automate download
    # If train/test split exists, use that
    if os.path.exists(os.path.join(get_data_source(), name, "train_data.npz")):
        train_data_all = np.load(os.path.join(get_data_source(), name, "train_data.npz"))
        train_data = train_data_all["data"]
        train_labels = train_data_all["labels"]
        test_data_all = np.load(os.path.join(get_data_source(), name, "test_data.npz"))
        test_data = test_data_all["data"]
        test_labels = test_data_all["labels"]
    else:
        # Else, create one
        df_path = os.path.join(get_data_source(), "purchase100", "dataset_purchase")
        # Open file
        features = []
        labels = []
        print("Generating train-test data splits (one-time operation)!")
        with open(df_path, "r") as f:
            for line in f:
                line = line.strip()
                splitted_line = line.split(",")
                labels.append(int(splitted_line[0]) - 1)
                features.append(list(map(int, splitted_line[1:])))
        data = np.array(features)
        labels = np.array(labels)
        # Shuffle indices
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        train_indices, test_indices = indices[:num_train], indices[num_train:]

        # Make sure directory exists
        os.makedirs(os.path.join(get_data_source(), name), exist_ok=True)

        # Save train and test data
        train_data = data[train_indices]
        train_labels = labels[train_indices]
        np.savez(os.path.join(get_data_source(), name, "train_data"), data=train_data, labels=train_labels)
        test_data = data[test_indices]
        test_labels = labels[test_indices]
        np.savez(os.path.join(get_data_source(), name, "test_data"), data=test_data, labels=test_labels)

    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)

    return (train_data, train_labels), (test_data, test_labels)


class Purchase100(Dataset):
    def __init__(self, augment: bool = True):
        num_train = 80_000
        # Double to account for pkeep=0.5
        (train_data, train_labels), (test_data, test_labels) = load_purchase100_file("purchase100", num_train * 2)

        # Convert to torch datasets
        train_data = ch.from_numpy(train_data)
        train_labels = ch.from_numpy(train_labels)
        test_data = ch.from_numpy(test_data)
        test_labels = ch.from_numpy(test_labels)
        train_data = ch.utils.data.TensorDataset(train_data, train_labels)
        test_data = ch.utils.data.TensorDataset(test_data, test_labels)

        super().__init__("Purchase100", train_data, test_data, num_classes=100)


class Purchase100Small(Dataset):
    def __init__(self, augment: bool = True):
        num_train = 25_000
        # Double to account for pkeep=0.5
        (train_data, train_labels), (test_data, test_labels) = load_purchase100_file("purchase100_s", num_train * 2)

        # Convert to torch datasets
        train_data = ch.from_numpy(train_data)
        train_labels = ch.from_numpy(train_labels)
        test_data = ch.from_numpy(test_data)
        test_labels = ch.from_numpy(test_labels)
        train_data = ch.utils.data.TensorDataset(train_data, train_labels)
        test_data = ch.utils.data.TensorDataset(test_data, test_labels)

        super().__init__("Purchase100Small", train_data, test_data, num_classes=100)
