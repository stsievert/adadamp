from __future__ import print_function
from types import SimpleNamespace
from typing import Callable, Dict, Any, Union, Any, Union, Optional
from copy import copy, deepcopy
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
from distributed import Client, LocalCluster

from adadamp._dist import gradient

"""
  - Look at dask scaling number of workers to grow with batch size
  (use example whre batch size grows by 1 each iteration)
  - Look at improving speed of train function by having it load data in train
    - Load before train and pass futures to the train function 
    - Pass a client to train and pass it futures

  Goal: Give every worker all the data, compute gradients for these indivies for different
        workers
"""

class Net(nn.Module):
    """
    Net for classification of FashionMINST dataset
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
    """
    Gets FashionMINWT test and train data
    """
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


def _batch_size(model_updates: int, base: int = 64, increase: float = 0.1) -> int:
    """
    Computes next batch size
    """
    return int(np.ceil(base + increase * model_updates))


def _get_gradients(client, model_future, train_data_future, train_lbl_future, batch_size, n_workers, verbose): 
    """
    Calculates the gradients at a given state
    """
    # get batches for each worker to compute
    rng = np.random.RandomState()
    idx = rng.choice(len(train_set), size=batch_size)
    worker_idxs = np.array_split(idx, n_workers)
    # compute gradients
    start = time.time()
    grads = [
        client.submit(
            gradient,
            train_data_future,
            train_lbl_future,
            model=model_future,
            loss=F.nll_loss,
            idx=worker_idx,
        )
        for worker_idx in worker_idxs
    ]
    if verbose:
        print("\t Computed gradient in {:.3f} seconds".format(time.time() - start))
    return client.gather(grads)

def test(args, model, device, test_loader, verbose=True):
    # Small modification of PyTorch's MNIST example
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    acc = correct / len(test_loader.dataset)
    if verbose:
        msg = "\nTest set: loss_avg={:.3f}, accuracy=({:.1f}%)\n"
        print(msg.format(test_loss, 100.0 * acc))
    return {"acc": acc, "loss": test_loss}

def train_model(model, train_set, kwargs):
    """
    Trains model using an adaptive batch size
    """
    # create client and scatter data
    print("Creating Dask client...")
    start = time.time()
    n_workers = 1
    cluster = LocalCluster(n_workers=n_workers)
    client = Client(cluster, serializers=['pickle'])
    print("=== Completed in {:.3f} seconds".format(time.time() - start))

    # scatter data ahead of time
    print("Sending data to workers...")
    start = time.time()
    train_data_future = [client.scatter(pt[0], broadcast=True) for pt in train_set] # client.scatter(train_data)
    train_lbl_future = [client.scatter(pt[1], broadcast=True) for pt in train_set]  #client.scatter(train_lbl)
    print("=== Completed in {:.3f} seconds".format(time.time() - start))

    # run SGF, updating BS when needed
    print("Running SDG on model for {} epochs...".format(kwargs["epochs"]))
    opt = optim.SGD(model.parameters(), lr=kwargs["lr"])

    # run gradients for however many grads
    for model_updates in range(kwargs["epochs"]):
        # track when to update batch size
        if model_updates % kwargs["dwell"] == 0:
            print("Updating batch size")
            bs = _batch_size(kwargs["initial_batch_size"], model_updates, kwargs["batch_growth_rate"])
            n_workers = bs // 16
            # we want the works to scale with the batch size exactly
            if cluster.n_workers != n_workers:
                client.scale(n_workers)
        # use the model to get the next grad step
        model_future = client.scatter(copy(model), broadcast=True)
        grads = _get_gradients(client, model_future, train_data_future, train_lbl_future, batch_size=bs, n_workers=n_workers, verbose=True)  # a call to Dask
        # update SGD
        opt.zero_grad()
        num_data = sum(info["_num_data"] for info in grads)
        assert num_data == batch_size
        for name, param in model.named_parameters():
            grad = sum(grad[name] for grad in grads)
            param.grad = grad / num_data



if __name__ == "__main__":
    # from to-joe
    kwargs = {
        "lr":0.0433062924,
        "batch_growth_rate": 0.3486433523,
        "dwell": 100,
        "max_batch_size": 1024,
        "initial_batch_size": 24,
        "epochs": 20_000,
    }
    model = Net()
    train_set, test_set = _get_fashionmnist()
    train_model(model, train_set, kwargs)
