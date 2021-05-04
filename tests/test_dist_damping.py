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
from adadamp import DaskBaseDamper, DaskClassifier, DaskRegressor
from adadamp.dampers import GeoDamp
from sklearn.datasets import make_regression


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
        "optimizer__lr": 1e-3,
        "device": device,
    }

    est1 = DaskBaseDamper(**kwargs).initialize()
    score1 = {"initial": est1.score(X, y)}
    est1.partial_fit(X, y)

    # run partial fit many times. Each partial fit does one update
    est2 = DaskBaseDamper(**kwargs).initialize()
    score2 = {"initial": est2.score(X, y)}
    for k in range(n_updates):
        idx = np.arange(batch_size * k, batch_size * (k + 1)).astype(int)
        est2.partial_fit(X[idx], y[idx])
    score1["final"] = est1.score(X, y)
    score2["final"] = est2.score(X, y)
    assert np.allclose(score1["initial"], score2["initial"])
    assert max(score1["final"], score2["final"]) < score1["initial"] / 4


def test_max_batch_size():
    batch_size = 128
    n_updates = 5

    n = n_updates * batch_size
    d = 10

    X, y = _random_dataset(n, d)
    y = y.reshape(-1, 1)

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"

    est = DaskBaseDamper(
        module=LinearNet,
        module__d=d,
        loss=nn.MSELoss,
        max_batch_size=128,
        batch_size=256,
        optimizer=optim.Adadelta,
        lr=1,
    )
    est.partial_fit(X, y)
    assert est.meta_["lr_"] == 0.5
    assert est.meta_["batch_size_"] == 128


def test_max_batch_size():
    batch_size = 128
    n_updates = 5

    n = n_updates * batch_size
    d = 10

    X, y = _random_dataset(n, d)
    #  y = y.reshape(-1, 1)

    device = "cpu" if not torch.cuda.is_available() else "cuda:0"

    bs = 256
    mbs = bs * 4
    lr = 1
    est = DaskBaseDamper(
        module=LinearNet,
        module__d=d,
        loss=nn.MSELoss,
        max_batch_size=mbs,
        batch_size=bs,
        optimizer=optim.Adadelta,
        lr=lr,
    )
    for i in range(1, 8):
        d = est.batch_size * i
        est.damping_ = d
        est.partial_fit(X, y)
        assert est.meta_["batch_size_"] == min(mbs, bs * i)
        factor = est.meta_["batch_size_"] / est.meta_["damping_"]
        assert est.meta_["lr_"] == factor * lr

        noise_level = est.meta_["batch_size_"] / est.meta_["lr_"]
        assert np.allclose(noise_level, 256 * i)


class HiddenLayer(nn.Module):
    def __init__(self, features=4, hidden=2, out=1):
        super().__init__()
        self.hidden = nn.Linear(features, hidden)
        self.out = nn.Linear(hidden, out)

    def forward(self, x, *args, **kwargs):
        ir = F.relu(self.hidden(x))
        return self.out(ir)


def test_geodamp():
    est = DaskRegressor(
        module=HiddenLayer,
        module__features=10,
        optimizer=optim.Adadelta,
        optimizer__weight_decay=1e-7,
        max_epochs=10,
    )
    est.set_params(batch_size=GeoDamp, batch_size__delay=60, batch_size__factor=5)
    X, y = make_regression(n_features=10)
    X = torch.from_numpy(X.astype("float32"))
    y = torch.from_numpy(y.astype("float32")).reshape(-1, 1)
    est.fit(X, y)
    score = est.score(X, y)
    assert -1 < score



if __name__ == "__main__":
    client = Client()
    client.run(_prep)
    test_geodamp()
    test_dask_damper_updates()
    test_max_batch_size()
