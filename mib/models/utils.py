from mib.models.wide_resnet import Wide_ResNet
from torch import nn


MODEL_MAPPING = {
    "wide_resnet_28_1": {
        "model": (Wide_ResNet, (28, 1)),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    "wide_resnet_28_2": {
        "model": (Wide_ResNet, (28, 2)),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    "wide_resnet_28_10": {
        "model": (Wide_ResNet, (28, 10)),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
}


def get_model(name: str, n_classes: int):
    if name not in MODEL_MAPPING:
        raise ValueError(f"Model {name} not found.")
    model_info = MODEL_MAPPING[name]
    model_class, params = model_info["model"]
    criterion = model_info["criterion"]
    hparams = model_info["hparams"]
    return model_class(*params, n_classes), criterion, hparams
