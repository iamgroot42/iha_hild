from mib.dataset.cifar10 import CIFAR10
from mib.dataset.cifar100 import CIFAR100
from mib.dataset.purchase100 import Purchase100
from mib.dataset.mnist import MNIST17, MNISTOdd

DATASET_MAPPING = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "purchase100": Purchase100,
    "mnist17": MNIST17,
    "mnistodd": MNISTOdd
}


def get_dataset(name: str):
    if name not in DATASET_MAPPING:
        raise ValueError(f"Dataset {name} not found.")
    model_class = DATASET_MAPPING[name]
    return model_class
