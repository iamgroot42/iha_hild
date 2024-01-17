from mib.dataset.cifar10 import CIFAR10

DATASET_MAPPING = {
    "cifar10": CIFAR10
}


def get_dataset(name: str):
    if name not in DATASET_MAPPING:
        raise ValueError(f"Dataset {name} not found.")
    model_class = DATASET_MAPPING[name]
    return model_class
