from torchvision import transforms, models, datasets
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import torchvision
import torch
import torch.nn as nn
from packaging import version

from adadamp.damping import BaseDamper
from adadamp.utils import _get_resnet18

def test_resnet18():
    transform_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

    model = _get_resnet18()
    _dir = "_traindata/cifar10/"
    train_set = datasets.CIFAR10(
        _dir, train=True, transform=Compose(transform_train), download=True,
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    damper = BaseDamper(model, train_set, opt, initial_batch_size=4)

    # Make sure data is fed into model correctly
    damper.step()
    assert damper._meta["model_updates"] == 1  # sanity check
