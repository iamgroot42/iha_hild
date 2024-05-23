from torchvision import transforms
import torchvision
import torch as ch
from mib.utils import get_data_source

from mib.dataset.base import Dataset


class FMNISTUpperFoot(Dataset):
    def __init__(self, augment: bool = True):
        standard_image_transform = transforms.ToTensor()

        data_root = get_data_source()
        train_data = torchvision.datasets.FashionMNIST(
            root=data_root,
            train=True,
            download=True,
            transform=standard_image_transform,
        )
        test_data = torchvision.datasets.FashionMNIST(
            root=data_root,
            train=False,
            download=True,
            transform=standard_image_transform,
        )

        upperwear_classes = ch.tensor([0, 2, 6])
        footwear_classes = ch.tensor([5, 7, 9])
        train_data.targets = train_data.targets.clone().detach()
        test_data.targets = test_data.targets.clone().detach()

        upperwear_train_classes = ch.isin(train_data.targets, upperwear_classes)
        upperwear_test_classes  = ch.isin(test_data.targets, upperwear_classes)
        footwear_train_classes  = ch.isin(train_data.targets, footwear_classes)
        footwear_test_classes   = ch.isin(test_data.targets, footwear_classes)

        subset_train = ch.logical_or(upperwear_train_classes, footwear_train_classes)
        subset_test = ch.logical_or(upperwear_test_classes, footwear_test_classes)
        # Create Tensordata out of this
        train_data.data = train_data.data[subset_train].clone().float() / 255.0
        test_data.data = test_data.data[subset_test].clone().float() / 255.0
        # Discard all other classes
        train_data.targets = train_data.targets[subset_train].clone()
        test_data.targets = test_data.targets[subset_test].clone()

        # Upperwear: label 0, Footwear: label 1
        # For train
        train_data.targets = train_data.targets % 2
        test_data.targets = test_data.targets % 2

        # Flatten input data
        train_data.data = train_data.data.view(train_data.data.size(0), -1)
        test_data.data = test_data.data.view(test_data.data.size(0), -1)
        # Create TensorDataset out of this
        train_data = ch.utils.data.TensorDataset(train_data.data, train_data.targets)
        test_data = ch.utils.data.TensorDataset(test_data.data, test_data.targets)
        super().__init__("FMNISTUpperFoot", train_data, test_data, num_classes=1)


class FMNIST(Dataset):
    def __init__(self, augment: bool = True):
        standard_image_transform = transforms.ToTensor()

        data_root = get_data_source()
        train_data = torchvision.datasets.FashionMNIST(
            root=data_root,
            train=True,
            download=True,
            transform=standard_image_transform,
        )
        test_data = torchvision.datasets.FashionMNIST(
            root=data_root,
            train=False,
            download=True,
            transform=standard_image_transform,
        )

        train_data.targets = train_data.targets.clone().detach()
        test_data.targets = test_data.targets.clone().detach()

        # Create Tensordata out of this
        train_data.data = train_data.data.clone().float() / 255.0
        test_data.data = test_data.data.clone().float() / 255.0

        # Flatten input data
        train_data.data = train_data.data.view(train_data.data.size(0), -1)
        test_data.data = test_data.data.view(test_data.data.size(0), -1)
        # Create TensorDataset out of this
        train_data = ch.utils.data.TensorDataset(train_data.data, train_data.targets)
        test_data = ch.utils.data.TensorDataset(test_data.data, test_data.targets)
        super().__init__("FMNIST", train_data, test_data, num_classes=10)
