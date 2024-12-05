import torchvision
from torchvision import models
import torch.nn as nn
from packaging import version

def _get_resnet18():
    model = models.resnet18()
    # See https://github.com/pytorch/vision/issues/696
    if version.parse(torchvision.__version__) < version.parse("v0.3.0"):
        model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
