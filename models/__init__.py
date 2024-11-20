from .model import self_net
from .baseline.unet import UNet
from .baseline.unetplusplus import UNetplusplus
from .baseline.unet3plus import UNet3plus

__all__ = [
    "self_net",
    "UNet",
    "UNetplusplus",
    "UNet3plus",
]

def load_model(model_name, n_channels, num_classes):
    if model_name == "self_net":
        model = self_net(n_channels=n_channels, n_classes=num_classes)
    elif model_name == "UNet":
        model = UNet(n_channels=n_channels, n_classes=num_classes)
    elif model_name == "UNetplusplus":
        model = UNetplusplus(input_channels=n_channels, num_classes=num_classes)
    elif model_name == "UNet3plus":
        model = UNet3plus(in_channels=n_channels, n_classes=num_classes)
    else:
        raise ValueError("Invalid model name")
    return model
