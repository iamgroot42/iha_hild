from mib.models.wide_resnet import Wide_ResNet
from mib.models.nasnet import NasNetA
from mib.models.shufflenetv2 import ShuffleNetV2
from mib.models.mlp import MLP, MLPQuadLoss
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
    "nasnet": {
        "model": (NasNetA, (4, 2, 44, 44)),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    "shufflenet_v2_s": {
        "model": (ShuffleNetV2, (1,)),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    "shufflenet_v2_m": {
        "model": (ShuffleNetV2, (1.5,)),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    },
    "mlp4": {
        "model": (
            MLP,
            (
                600,
                [512, 256, 128, 64],
            ),
        ),
        "criterion": nn.CrossEntropyLoss(),
        # "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
        # "hparams": {"batch_size": 256, "learning_rate": 0.05, "epochs": 100},
        "hparams": {"batch_size": 256, "learning_rate": 0.05, "epochs": 120},
    },
    "mlp3": {
        "model": (
            MLP,
            (
                600,
                [128, 64, 32],
            ),
        ),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.05, "epochs": 120},
    },
    "mlp3_small": {
        "model": (
            MLP,
            (
                600,
                [32, 32, 8],
            ),
        ),
        "criterion": nn.CrossEntropyLoss(),
        "hparams": {"batch_size": 256, "learning_rate": 0.02, "epochs": 120},
    },
    "mlp_mnist17": {
        "model": (
            MLP,
            (
                784,
                [4],
            ),
        ),
        "criterion": nn.BCEWithLogitsLoss(),
        "hparams": {"batch_size": 128, "learning_rate": 0.001, "epochs": 100},
    },
    "mlp_mnistodd": {
        "model": (
            MLP,
            (
                784,
                [8],
            ),
        ),
        "criterion": nn.BCEWithLogitsLoss(),
        "hparams": {"batch_size": 128, "learning_rate": 0.01, "epochs": 100},
    },
    "mlp_mnistodd_mse": {
        "model": (MLPQuadLoss, (784, [8])),
        "criterion": nn.MSELoss(),
        "hparams": {"batch_size": 128, "learning_rate": 0.01, "epochs": 100},
    },
    # "mlp4_slow": {
    #     "model": (MLP, ([512, 256, 128, 64], )),
    #     "criterion": nn.CrossEntropyLoss(),
    #     "hparams": {"batch_size": 256, "learning_rate": 0.01, "epochs": 50},
    # }
    # "efficientnet_v2_s": {
    #     "model": (efficientnet_v2_s, ()),
    #     "criterion": nn.CrossEntropyLoss(),
    #     "hparams": {"batch_size": 256, "learning_rate": 0.1, "epochs": 100},
    # },
}


def get_model(name: str, n_classes: int):
    if name not in MODEL_MAPPING:
        raise ValueError(f"Model {name} not found.")
    model_info = MODEL_MAPPING[name]
    model_class, params = model_info["model"]
    criterion = model_info["criterion"]
    hparams = model_info["hparams"]
    return model_class(*params, n_classes), criterion, hparams
