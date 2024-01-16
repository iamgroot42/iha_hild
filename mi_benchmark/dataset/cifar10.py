from torchvision import transforms
import torchvision

from mi_benchmark.dataset.base import Dataset


class CIFAR10(Dataset):
    def __init__(self):
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
            root="/u/as9rw/work/auditing_mi", train=True, download=True, transform=transforms_train
        )
        test_data = torchvision.datasets.CIFAR10(
            root="/u/as9rw/work/auditing_mi", train=False, download=True, transform=transforms_test
        )
        super().__init__("CIFAR10", train_data, test_data, num_classes=10)
