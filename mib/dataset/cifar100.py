from torchvision import transforms
import torchvision
import torch as ch
from mib.utils import get_data_source

from mib.dataset.base import Dataset


class CIFAR100(Dataset):
    def get_augmented_input(self, x, y):
        flip_transform = transforms.RandomHorizontalFlip()

        # Standard image
        augmented = [x]

        # Un-normalize image
        x_unnorm = (x * 0.5) + 0.5
        for pad_h in [1, 2, 3, 4]:
            for pad_w in [1, 2, 3, 4]:
                # Add cropped image
                aug = transforms.RandomCrop(32, padding=(pad_h, pad_w))(x_unnorm)
                augmented.append((aug - 0.5) / 0.5)

                # And cropped + flipped image
                aug_flip = flip_transform(aug)
                augmented.append((aug_flip - 0.5) / 0.5)

        return ch.cat(augmented, 0)

    def train_transforms(self):
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

    def test_transforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

    def __init__(self, augment: bool = True):
        transforms_train = self.train_transforms()
        transforms_test = self.test_transforms()

        data_root = get_data_source()
        train_data = torchvision.datasets.CIFAR100(
            root=data_root,
            train=True,
            download=True,
            transform=transforms_train if augment else transforms_test,
        )
        test_data = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=transforms_test
        )
        super().__init__("CIFAR100", train_data, test_data, num_classes=100)
