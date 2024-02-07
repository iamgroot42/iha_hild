from torchvision import transforms
import torchvision
from mib.utils import get_data_source

from mib.dataset.base import Dataset


class CIFAR10(Dataset):
    def __init__(self, augment: bool = True):
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

        data_root = get_data_source()
        train_data = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transforms_train if augment else transforms_test
        )
        test_data = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transforms_test
        )
        super().__init__("CIFAR10", train_data, test_data, num_classes=10)
