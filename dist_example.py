from __future__ import print_function
from types import SimpleNamespace
from typing import Callable, Dict, Any, Union, Any, Union, Optional
from copy import copy, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
from distributed import Client

from adadamp._dist import gradient


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_batch_size(model_updates: int, base: int = 64, increase: float = 0.1) -> int:
    return int(np.ceil(base + increase * model_updates))


def train(
    model: nn.Module,
    device: torch.device,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    verbose: Optional[Union[bool, float]] = 0.1,
) -> Dict[str, Union[int, float]]:
    # The model will be stored on the master node. The workers will be assigned
    # to calculate the gradients. When they return the gradients, the master
    # node will iterate on the model, and repeat the loop (again asking workers
    # to compute the gradient).
    model.train()
    rng = np.random.RandomState(abs(hash(str(epoch))) % (2 ** 31))

    if not hasattr(model, "_updates"):
        model._updates = 0
        model._num_eg_processed = 0
    if isinstance(verbose, bool):
        verbose = 0.1

    _num_eg_print = -1
    _start_eg = copy(model._num_eg_processed)

    client = Client()
    future_inputs = client.scatter(inputs)
    future_targets = client.scatter(targets)
    future_device = client.scatter(device)

    while True:
        # Let's set the number of workers to be static for now.
        # This should grow as this optimization proceeds, either
        # explicitly in this code or implicitly with Dask adaptive
        # scaling (https://docs.dask.org/en/latest/setup/adaptive.html).
        # If implicit, it should grow with the batch size so each worker
        # processes a fixed number of gradients (or close to a fixed
        # number)
        n_workers = 16

        # Give all the data to each worker -- let's focus on
        # expensive computation with small data for now (it can easily be
        # generalized).
        batch_size = get_batch_size(model._updates, base=64, increase=0.05)
        idx = rng.choice(len(train_set), size=batch_size)
        worker_idxs = np.array_split(idx, n_workers)

        # Distribute the computation of the gradient. In practice this will
        # mean (say) 4 GPUs to accelerate the gradient computation. Right now
        # for ease it's a small network that doesn't need much acceleration.
        start = time.time()
        grads = [
            client.submit(
                gradient,
                future_inputs,
                future_targets,
                model=deepcopy(model),
                loss=F.nll_loss,
                device=device,
                idx=worker_idx,
            )
            for worker_idx in worker_idxs
        ]
        grads = client.gather(grads)
        print("Computed gradients in {:.5f} seconds".format(time.time() - start))
        # still taking 3+ seconds to get gradients. I assume this is because I dwas not scattering
        # the model, but I tried for a while and could not get that to work either

        # The gradients have been calculated. Now, let's make sure
        # ``optimizer`` can see the gradients.
        optimizer.zero_grad()
        num_data = sum(info["_num_data"] for info in grads)
        assert num_data == batch_size
        for name, param in model.named_parameters():
            grad = sum(grad[name] for grad in grads)
            param.grad = grad / num_data

        # Take an optimization step
        optimizer.step()

        # From here to the end of the loop is metadata tracking: how many
        # updates/epochs (or passes through the data), what's the loss, etc
        model._updates += 1
        model._num_eg_processed += num_data

        epochs = model._num_eg_processed / len(train_set)
        epochs_since_print = epochs - (_num_eg_print / len(targets))
        loss_avg = sum(grad["_loss"] for grad in grads) / num_data
        if _num_eg_print == -1 or epochs_since_print > verbose:
            _num_eg_print = model._num_eg_processed
            print(
                "epoch={:0.3f}, updates={}, loss={:0.3f}".format(
                    epochs, model._updates, loss_avg
                )
            )
        if model._num_eg_processed >= len(targets) + _start_eg:
            break
    return {"num_data": model._num_eg_processed, "batch_loss": loss_avg}


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


if __name__ == "__main__":
    # Training settings
    args = SimpleNamespace(
        batch_size=64,
        test_batch_size=1000,
        epochs=14,
        gamma=0.7,
        lr=1.0,
        no_cuda=False,
        save_model=False,
        seed=1,
    )
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST(
        "../data", train=True, download=True, transform=transform,
    )
    # For quicker debugging uncomment this line
    #  train_set, _ = torch.utils.data.random_split(train_set, [2000, len(train_set) - 2000])

    _train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=10_000, shuffle=False, **kwargs
    )

    # _train_loader's shuffle=False is importnat for these lines, otherwise
    # we're looping twice over the data, and it gets shuffled in- the meantime
    inputs = torch.cat([input.clone() for input, _ in _train_loader])
    targets = torch.cat([target.clone() for _, target in _train_loader])

    test_set = datasets.MNIST("../data", train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    data = []
    for epoch in range(1, args.epochs + 1):
        r = train(model, device, inputs, targets, optimizer, epoch, verbose=0.02)
        datum = test(args, model, device, test_loader)
        data.append(datum)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
