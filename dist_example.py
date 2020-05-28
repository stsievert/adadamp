from __future__ import print_function
from types import SimpleNamespace
from typing import Callable, Dict, Any, Union, Any, Union, Optional
from copy import copy, deepcopy
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose
from dist_example_model import Net
import torch
import torch.optim as optim
import sys
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
from distributed import Client, LocalCluster
from adadamp._dist import gradient
from functools import partial, lru_cache
import pandas as pd


@lru_cache()
def _get_fashionmnist():
    """
    Gets FashionMINWT test and train data

    Return
    ------
    Returns a tuple with PyTorch data loaders for both the train and test set
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


def _get_gradients(client, model_future, n, idx, batch_size, n_workers):
    """
    Calculates the gradients at a given state. This function is
    mainly in charge of sending the gradient calculation off to the
    dask workers, the hard work is done in _dist.gradient()

    Note:
    - In this implementation, each worker manually grabs the data
      and caches it themselves. This is a work around for a dask bug
    - See the doc string of _get_fashionminst in _dist.py for more info

    Parameters
    ----------

    client : distributed.Client()
        Dask client used to scheduele workers. Assumes client
        is backed by some cluster (ie LocalCluster or equiv)

    model_future : distributed.future (?)
        Future object of the model to get gradients for

    n : int
        number of items in training set
    
    idx: int
        Current epoch, used to set up random state

    batch_size: int
        Current batch size

    n_workers: int
        Number of workers current running

    """
    # pass a seed here
    rng = np.random.RandomState(
        seed=idx
    )  # Deterministic way to generate random numbers
    idx = rng.choice(n, size=batch_size)  # size random indexes
    worker_idxs = np.array_split(idx, n_workers)

    ## This is a dirty hack. See https://github.com/dask/distributed/issues/3807
    ## for more detail
    #  train_set, test_set = _get_fashionmnist()
    #  train_set = client.scatter(train_set, broadcast=True)
    train_set = None

    # TODO: Transition to use map rather than submit
    #       -- see https://github.com/stsievert/adadamp/pull/2#discussion_r430012263
    #
    # Distribute the computation of the gradient. In practice this will
    # mean (say) 4 GPUs to accelerate the gradient computation. Right now
    # for ease it's a small network that doesn't need much acceleration. 
    grads = [
        client.submit(
            gradient, train_set, model=model_future, loss=F.nll_loss, idx=worker_idx,
        )
        for worker_idx in worker_idxs
    ]
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

    Parameters
    ----------
    model : nn.Model
        Pytorch model to train

    train_set : torch.Tensor
        Data to train model on

    kwargs : dic
        Dictionary of key arguments, expects following keys
        - lr: learning rate
        - batch_growth_rate: batch growth rate
        - dwell: update batch size every dwell epochs
        - max_batch_size: larges possible batch size
        - grads_per_worker: the amount of gradients each worker will compute
        - initial_batch_size: initial batch size
        - initial_workers: how many workers to start with
        - epochs: how many loops to train for
        - save_freq: save every N epochs
        - log_suffix: appended to end of log save files
    """
    # create client and scatter data
    print("Creating Dask client...")
    start = time.time()
    n_workers = kwargs["initial_workers"]
    # LocalCluster is just a stand in for now, ideally 
    # this would be some network cluster when running the experimenets
    cluster = LocalCluster(n_workers=n_workers)
    client = Client(cluster)

    # scatter data ahead of time
    client_init = time.time() - start

    # run SGF, updating BS when needed
    print("Running SDG on model for {} epochs...".format(kwargs["epochs"]))
    opt = optim.SGD(model.parameters(), lr=kwargs["lr"])

    # init our metric tracking
    metrics = []  # will add pandas datafram with keys: loop_time, grad_time
    start_time = time.time()

    # run gradients for however many grads
    bs = -1
    log_suffix = kwargs["log_suffix"]
    for model_updates in range(kwargs["epochs"]):

        loop_start = time.time()

        # track when to update batch size
        if model_updates % kwargs["dwell"] == 0:
            # Give all the data to each worker -- let's focus on
            # expensive computation with small data for now (it can easily be
            # generalized).
            bs = _batch_size(
                kwargs["initial_batch_size"], model_updates, kwargs["batch_growth_rate"]
            )
            n_workers = max(kwargs["initial_workers"], bs // kwargs["grads_per_worker"])
            print(
                "Updating batch size to {}, computing across {} workers".format(
                    bs, n_workers
                )
            )
            # we want the works to scale with the batch size exactly
            if cluster.workers != n_workers:
                cluster.scale(n_workers)
        # use the model to get the next grad step
        new_model = copy(model)
        # this removes training spefic data from the model
        new_model.eval()
        model_future = client.scatter(new_model, broadcast=True)
        # compute grads
        grad_start = time.time()
        grads = _get_gradients(
            client,
            model_future,
            n=len(train_set),
            idx=model_updates,
            batch_size=bs,
            n_workers=n_workers,
        )  # a call to Dask
        grad_time = time.time() - grad_start
        # update SGD
        opt.zero_grad()
        num_data = sum(info["_num_data"] for info in grads)
        assert num_data == bs
        # aggregate and update the gradients
        for name, param in model.named_parameters():
            grad = sum(grad[name] for grad in grads)
            # --- move the averaging to get_gradiations
            param.grad = grad / num_data

        # update metrics!
        metrics.append(
            {
                "grad_time": grad_time,
                "bs": bs,
                "n_workers": n_workers,
                "loop_time": time.time() - loop_start,
                "kwargs": kwargs,
                "idx": model_updates,
            }
        )

        if model_updates % kwargs["save_freq"] == 0:
            print("Saving metrics to disk")
            pd.DataFrame(metrics).to_csv("loop_metrics" + log_suffix + ".csv")

    # have last entry have overall data: total time, client init time, send to data to clinet time
    overview_metrics = pd.DataFrame.from_dict(
        {"train_time": time.time() - start_time, "client_init": client_init}
    )
    loop_metrics = pd.DataFrame(metrics)
    overview_metrics.to_csv("overview_metrics" + log_suffix + ".csv")
    loop_metrics.to_csv("loop_metrics" + log_suffix + ".csv")


if __name__ == "__main__":
    # from to-joe
    kwargs = {
        "lr": 0.0433062924,
        "batch_growth_rate": 0.3486433523,
        "dwell": 100,
        "max_batch_size": 1024,
        "grads_per_worker": 64,
        "initial_batch_size": 24,
        "initial_workers": 3,
        "epochs": 20_000,
        "save_freq": 10,
        "log_suffix": "_t1",
    }
    model = Net()
    train_set, test_set = _get_fashionmnist()
    train_model(model, train_set, kwargs)
