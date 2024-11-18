import torch
import torch_directml
from model import self_net
from unet import UNet
from unetplusplus import UNetplusplus
from unet3plus import UNet3plus
from unetv2 import UNetV2

__all__ = [
    "self_net",
    "UNet",
    "UNetplusplus",
    "UNet3plus",
    "UNetV2",
]

def load_model(model_name, n_channels, num_classes):
    if model_name == "self_net":
        model = self_net(n_channels, num_classes)
    elif model_name == "UNet":
        model = UNet(n_channels=n_channels, n_classes=num_classes)
    elif model_name == "UNetplusplus":
        model = UNetplusplus(n_channels, num_classes=num_classes)
    elif model_name == "UNet3plus":
        model = UNet3plus(n_channels, n_classes=num_classes)
    elif model_name == "UNetV2":
        model = UNetV2(n_channels, n_classes=num_classes)
    else:
        raise ValueError("Invalid model name")

    try:
        if torch.cuda.is_available():
            model = model.to(device=torch.device("cuda"))
        else:
            model = model.to(device=torch_directml.device())
    except:
        pass

    return model
