from __future__ import print_function

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from distributed import Client
from distributed.utils_test import gen_cluster
from sklearn.utils import check_random_state

from test_sklearn_interface import Net, X, y
from adadamp import DaskBaseDamper, DaskClassifier


class LinearNet(nn.Module):
    def __init__(self, d=10, out=1):
        super().__init__()
        self.d = d
        self.linear = nn.Linear(d, out)

    def forward(self, x):
        return self.linear(x).reshape(-1)


def _random_dataset(n, d, random_state=None):
    rng = check_random_state(random_state)
    X = rng.uniform(size=(n, d)).astype("float32")
    y = rng.uniform(size=n).astype("float32")
    return X, y


def _prep():
    from distributed.protocol import torch


def test_dask_damper_updates():
    batch_size = 128
    n_updates = 5

    n = n_updates * batch_size
    d = 10

    X, y = _random_dataset(n, d)
    y = y.reshape(-1, 1)

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"
    kwargs = {
        "batch_size": batch_size,
        "max_epochs": 1,
        "random_state": 42,
        "module": LinearNet,
        "module__d": d,
        "weight_decay": 1e-5,
        "loss": nn.MSELoss,
        "optimizer": optim.SGD,
        "optimizer__lr": 0.1,
        "device": device,
    }

    est1 = DaskBaseDamper(**kwargs)
    est1.partial_fit(X, y)
    assert est1.meta_["n_weight_changes"] == est1.meta_["n_updates"] == n_updates

    # run partial fit many times. Each partial fit does one update
    est2 = DaskBaseDamper(**kwargs)
    for k in range(n_updates):
        idx = np.arange(batch_size * k, batch_size * (k + 1)).astype(int)
        est2.partial_fit(X[idx], y[idx])
    assert est2.meta_["n_weight_changes"] == n_updates


if __name__ == "__main__":
    client = Client()
    client.run(_prep)
    test_dask_damper_updates()
