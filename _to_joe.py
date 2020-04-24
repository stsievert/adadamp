import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import numpy as np

from adadamp import PadaDamp


class Net(nn.Module):
    """
    111k params, ~150s/epoch
    """

    def __init__(self):
        super(Net, self).__init__()
        self.hidden_size = 100
        self.final_convs = 100
        self.conv1 = nn.Conv2d(1, 30, 5, stride=1)
        self.conv2 = nn.Conv2d(30, 60, 5, stride=1)
        self.conv3 = nn.Conv2d(60, self.final_convs, 3, stride=1)
        self.fc1 = nn.Linear(1 * 1 * self.final_convs, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 1 * 1 * self.final_convs)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _get_fashionmnist():
    transform_train = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ]
    transform_test = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    _dir = "_traindata/fashionmnist/"
    train_set = FashionMNIST(
        _dir, train=True, transform=Compose(transform_train), download=True,
    )
    test_set = FashionMNIST(_dir, train=False, transform=Compose(transform_test))
    return train_set, test_set

def _batch_size(base: int, model_updates: int, rate: float) -> int:
    """
    Find the batch size for a give number of model updates. The model is
    updated everytime `opt.step` is called.

    Parameters
    ----------
    base : int
        The initial batch size
    model_updates : int
        The number of model updates
    rate : float
        The rate at which to increase the batch size. At any number of model
        updates, the batch size is given by
        ``int(model_updates*rate + base)``

    Returns
    -------
    bs : int
        The batch size
    """
    return int(np.ceil(base + model_updates*rate))

if __name__ == "__main__":
    kwargs = {
        "lr":0.0433062924,
        "batch_growth_rate": 0.3486433523,
        "dwell": 100,
        "max_batch_size": 1024,
        "initial_batch_size": 24,
    }
    model = Net()
    train_set, test_set = _get_fashionmnist()
    opt = optim.SGD(model.parameters(), lr=kwargs["lr"])

    ## If I were running locally...
    # opt = PadaDamp(
    #     model,
    #     train_set,
    #     opt,
    #     **kwargs,
    # )

    ## But I'm not, so here's some psuedo-code.
    ## adadamp/damping.py might have more detail
    for model_updates in range(20_000):
        if model_updates % kwargs["dwell"] == 0:
            bs = _batch_size(kwargs["initial_batch_size"], model_updates, kwargs["batch_growth_rate"])
        grads = _get_gradients(model, train_set, batch_size=bs)  # a call to Dask
        opt.step()
