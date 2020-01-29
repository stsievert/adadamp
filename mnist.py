# modified from [1]
# [1]:https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
from types import SimpleNamespace
from typing import Any, Dict, List
import hashlib
import pickle
from pprint import pprint
import itertools

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import damping
import experiment


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 26 ** 2, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


def stable_hash(w: bytes) -> bytes:
    h = hashlib.sha256(w)
    return h.hexdigest()


def ident(args: dict) -> str:
    ordered_keys = sorted(list(args.keys()))
    v = [
        (k, args[k] if not isinstance(args[k], dict) else ident(args[k]))
        for k in ordered_keys
    ]
    return str(stable_hash(pickle.dumps(v)))


if __name__ == "__main__":
    args: Dict[str, Any] = {
        "initial_batch_size": 64,
        "epochs": 5,
        "verbose": 10,
        "lr": 1.0,
        "no_cuda": False,
        "seed": 1,
        "damper": "geodamp",
        "padarate": 0.01,  # padadamp
        "geofactor": 5,  # geodamp
        "geodelay": 2,  # geodamp. Number of epochs
        "max_batch_size": None,  # basedamper
    }
    args["ident"] = ident(args)
    args["prefix"] = "_data/2020-01-29/debug"
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args["seed"])

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    normalize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST("_data/", train=True, transform=normalize, download=True)
    test_set = datasets.MNIST("_data/", train=False, transform=normalize)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args["lr"])
    n_data = len(train_set)

    opt_args = [model, train_set, optimizer]
    opt_kwargs = {"initial_batch_size": args["initial_batch_size"]}
    if args["damper"].lower() == "padadamp":
        opt = damping.PadaDamp(*opt_args, rate=args["padarate"], **opt_kwargs)
    elif args["damper"].lower() == "geodamp":
        opt = damping.GeoDamp(
            *opt_args,
            geodelay=args["geodelay"],
            geofactor=args["geofactor"],
            **opt_kwargs
        )
    elif args["damper"].lower() == "adadamp":
        opt = damping.GeoDamp(*opt_args, **opt_kwargs)
    elif args["damper"].lower() == "none" or args["damper"] is None:
        opt = damping.BaseDamper(*opt_args, **opt_kwargs)
    else:
        raise ValueError("argument damper not recognized")

    data, train_data = experiment.run(
        model=model, opt=opt, train_set=train_set, test_set=test_set, args=args
    )
