from torchvision import transforms
import torchvision
import torch as ch
from mib.utils import get_data_source

from mib.dataset.base import Dataset


class MNIST17(Dataset):
    def __init__(self, augment: bool = True):
        standard_image_transform = transforms.ToTensor()

        data_root = get_data_source()
        train_data = torchvision.datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=standard_image_transform,
        )
        test_data = torchvision.datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=standard_image_transform,
        )

        # Take note of indices where label is 1 or 7
        train_data.targets = train_data.targets.clone().detach()
        test_data.targets = test_data.targets.clone().detach()
        subset_train = ch.logical_or(train_data.targets == 1, train_data.targets == 7)
        subset_test = ch.logical_or(test_data.targets == 1, test_data.targets == 7)
        # Create Tensordata out of this
        train_data.data = train_data.data[subset_train].clone().float() / 255.
        test_data.data = test_data.data[subset_test].clone().float() / 255.
        train_data.targets = train_data.targets[subset_train].clone()
        test_data.targets = test_data.targets[subset_test].clone()
        # Replace 1/7 labels with 1/0
        train_data.targets[train_data.targets != 1] = 0
        test_data.targets[test_data.targets != 1] = 0
        # Flatten input data
        train_data.data = train_data.data.view(train_data.data.size(0), -1)
        test_data.data = test_data.data.view(test_data.data.size(0), -1)
        # Create TensorDataset out of this
        train_data = ch.utils.data.TensorDataset(train_data.data, train_data.targets)
        test_data = ch.utils.data.TensorDataset(test_data.data, test_data.targets)
        super().__init__("MNIST17", train_data, test_data, num_classes=1)


class MNISTOdd(Dataset):
    def __init__(self, augment: bool = True):
        standard_image_transform = transforms.ToTensor()

        data_root = get_data_source()
        train_data = torchvision.datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=standard_image_transform,
        )
        test_data = torchvision.datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=standard_image_transform,
        )

        # Take note of indices where label is odd or eve
        train_data.targets = train_data.targets.clone().detach()
        test_data.targets = test_data.targets.clone().detach()
        # Create Tensordata out of this
        train_data.data = train_data.data.clone().float() / 255.0
        test_data.data = test_data.data.clone().float() / 255.0
        train_data.targets = train_data.targets.clone()
        test_data.targets = test_data.targets.clone()
        # Test for odd/eve
        train_data.targets = train_data.targets % 2
        test_data.targets = test_data.targets % 2
        # Flatten input data
        train_data.data = train_data.data.view(train_data.data.size(0), -1)
        test_data.data = test_data.data.view(test_data.data.size(0), -1)
        # Create TensorDataset out of this
        train_data = ch.utils.data.TensorDataset(train_data.data, train_data.targets)
        test_data = ch.utils.data.TensorDataset(test_data.data, test_data.targets)
        super().__init__("MNISTOdd", train_data, test_data, num_classes=1)
