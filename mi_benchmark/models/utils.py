from mi_benchmark.models.wide_resnet import Wide_ResNet

MODEL_MAPPING = {
    "wide_resnet_cifar": (Wide_ResNet, (28, 2, 10))
}

def get_model(name: str):
    if name not in MODEL_MAPPING:
        raise ValueError(f"Model {name} not found.")
    model_class, params = MODEL_MAPPING[name]
    return model_class(*params)
