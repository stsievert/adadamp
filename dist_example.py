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
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
from distributed import Client, LocalCluster
from adadamp._dist import gradient

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


def _get_gradients(client, model_future, train_set_future, n, idx, batch_size, n_workers, verbose): 
    """
    Calculates the gradients at a given state
    """
    # get batches for each worker to compute
    # HELP: What is this doing?

    # pass a seed here
    rng = np.random.RandomState(seed=idx) # Deterministic way to generate random numbers
    idx = rng.choice(n, size=batch_size) # size random indexes
    worker_idxs = np.array_split(idx, n_workers)
    # compute gradients
    start = time.time()
    grads = [
        client.submit(
            gradient,
            train_set_future,
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
    n_workers = kwargs["initial_workers"]
    cluster = LocalCluster(n_workers=n_workers)
    client = Client(cluster)
    # HELP: I think the workers must have the model, still errors when I upload thee model
    client.wait_for_workers(n_workers)
    client.upload_file('dist_example_model.py')

    # scatter data ahead of time
    train_set_future = client.scatter(train_set, broadcast=True)
    client_init = time.time() - start

    # run SGF, updating BS when needed
    print("Running SDG on model for {} epochs...".format(kwargs["epochs"]))
    opt = optim.SGD(model.parameters(), lr=kwargs["lr"])

    # init our metric tracking
    metrics = [] # will add pandas datafram with keys: loop_time, grad_time
    start_time = time.time()

    # run gradients for however many grads
    for model_updates in range(kwargs["epochs"]):

        loop_start = time.time()

        # track when to update batch size
        if model_updates % kwargs["dwell"] == 0:
            print("Updating batch size")
            bs = _batch_size(kwargs["initial_batch_size"], model_updates, kwargs["batch_growth_rate"])
            n_workers = max(kwargs["initial_workers"], bs // kwargs["grads_per_worker"])
            # we want the works to scale with the batch size exactly
            if cluster.workers != n_workers:
                cluster.scale(n_workers)
        # use the model to get the next grad step

        # HELP: Does this look correct for removing gradients?
        new_model = copy(model)
        new_model.zero_grad()

        model_future = client.scatter(copy(new_model), broadcast=True)
        grad_start = time.time()
        grads = _get_gradients(client, model_future, train_set_future, n=len(train_set), idx=model_updates, batch_size=bs, n_workers=n_workers, verbose=True)  # a call to Dask
        grad_time = time.time() - grad_start

        # update SGD
        opt.zero_grad()
        num_data = sum(info["_num_data"] for info in grads)
        assert num_data == batch_size
        for name, param in model.named_parameters():
            grad = sum(grad[name] for grad in grads)
            # --- move the averaging to get_gradiations
            param.grad = grad / num_data

        # update metrics!
        metrics.append(pd.DataFrame.from_dict({ 'grad_time': grad_time, 'loop_time': time.time() - loop_start, 'kwargs': kwargs, 'idx': model_updates }))

    # have last entry have overall data: total time, client init time, send to data to clinet time
    metrics.append(pd.DataFrame.from_dict({ 'train_time': time.time() - start_time, 'client_init': client_init }))

    print(metrics)




if __name__ == "__main__":
    # from to-joe
    kwargs = {
        "lr":0.0433062924,
        "batch_growth_rate": 0.3486433523,
        "dwell": 100,
        "max_batch_size": 1024,
        "grads_per_worker": 16,
        "initial_batch_size": 24,
        "initial_workers": 8,
        "epochs": 20_000,
    }
    model = Net()
    train_set, test_set = _get_fashionmnist()
    train_model(model, train_set, kwargs)
