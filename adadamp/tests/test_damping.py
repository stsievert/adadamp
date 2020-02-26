from __future__ import print_function
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
from sklearn.datasets import make_classification
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset

from adadamp import BaseDamper, GeoDamp, AdaDamp, PadaDamp
import adadamp.experiment as experiment


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output


@pytest.fixture
def dataset():
    X, y = make_classification(
        n_features=20, n_samples=128, n_classes=10, n_informative=10
    )
    return TensorDataset(
        torch.from_numpy(X.astype("float32")), torch.from_numpy(y.astype("int64"))
    )


@pytest.fixture
def model():
    return Net().to("cpu")


def test_basics(model, dataset):
    epochs = 14
    optimizer = optim.Adadelta(model.parameters(), lr=1)
    opt = BaseDamper(model, dataset, optimizer, initial_batch_size=8)

    data: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model, opt, meta, train_data = experiment.train(model, opt)
        data += train_data

    df = pd.DataFrame(data)
    assert (df.model_updates * df.batch_size == df.num_examples).all()
    assert df.epochs.max() <= epochs + 2
    eg_per_epoch = df.num_examples.diff().iloc[1:]
    len_dataset = df.len_dataset.iloc[1:]
    assert all((eg_per_epoch - len_dataset) <= df.batch_size.iloc[1:])


def test_geodamp(model, dataset):
    _opt = optim.Adadelta(model.parameters(), lr=1)
    opt = GeoDamp(
        model, dataset, _opt, initial_batch_size=1, dampingdelay=4, dampingfactor=2
    )
    data: List[Dict[str, Any]] = []
    # Specifically let GeoDamp train for at least one epoch
    for epoch in range(1, 16 + 1):
        model, opt, meta, _ = experiment.train(model, opt)
        data.append(opt.meta)
    df = pd.DataFrame(data)
    # Check to make sure it's exactly one epoch
    assert np.allclose(df.epochs, np.floor(df.epochs))
    counts = df.damping.value_counts()
    assert set(counts.index.astype(int)) == {1, 2, 4, 8}
    assert np.allclose(counts.unique(), 4)


def test_padadamp(model, dataset):
    _opt = optim.Adadelta(model.parameters(), lr=1)
    opt = PadaDamp(model, dataset, _opt, batch_growth_rate=1, initial_batch_size=1)
    data: List[Dict[str, Any]] = []
    for epoch in range(1, 16 + 1):
        model, opt, meta, train_data = experiment.train(model, opt)
        data += train_data
    df = pd.DataFrame(data)
    assert (df.damping == df.model_updates + 1).all()


def test_adadamp(model, dataset):
    init_bs = 8
    _opt = optim.SGD(model.parameters(), lr=0.500)
    opt = AdaDamp(model, dataset, _opt, initial_batch_size=init_bs)
    data: List[Dict[str, Any]] = []
    initial_loss = opt._get_loss()
    for epoch in range(5):
        model, opt, meta, train_data = experiment.train(model, opt)
        data += train_data
    df = pd.DataFrame(data)

    bs_hat = init_bs * df.loc[0, "_complete_loss"] / df._complete_loss
    bs_hat = bs_hat.values.astype(int) + 1
    bs = df.batch_size.values
    assert (bs == bs_hat).all()


def test_avg_loss(model, dataset):
    """
    Test that BaseDamper._get_loss returns mean loss regardless of how many
    points are sampled.
    """
    _opt = optim.Adadelta(model.parameters(), lr=1)
    opt = BaseDamper(model, dataset, _opt)
    for epoch in range(1, 16 + 1):
        model, opt, meta, _ = experiment.train(model, opt)
    loss = [
        {"loss": opt._get_loss(frac=frac), "frac": frac, "repeat": repeat}
        for frac in np.linspace(0.5, 0.99, num=5)
        for repeat in range(5)
    ]
    total_loss = opt._get_loss(frac=1)
    df = pd.DataFrame(loss)
    summary = df.pivot(index="frac", columns="repeat", values="loss")

    abs_error = np.abs(df.loss - total_loss)
    rel_error = abs_error / total_loss
    assert rel_error.max() <= 0.125
    assert np.percentile(rel_error, 50) <= 0.12
    assert 1.5 <= total_loss <= 2.2
    assert abs_error.max() <= 0.17


def test_get_params(model, dataset):
    _opt = optim.Adadelta(model.parameters(), lr=1)
    opt = BaseDamper(model, dataset, _opt)
    params = opt.get_params()

    opt2 = BaseDamper(model, dataset, _opt, **params)
    params2 = opt2.get_params()
    assert params == params2
    param_keys = {
        "device_type",
        "initial_batch_size",
        "loss_name",
        "max_batch_size",
    }
    meta_keys = {
        "model_updates",
        "num_examples",
        "batch_loss",
        "num_params",
        "len_dataset",
        "opt_param_lr",
        "opt_param_rho",
        "opt_param_eps",
        "opt_param_weight_decay",
        "initial_batch_size",
        "max_batch_size",
        "device_type",
        "loss_name",
        "epochs",
        "damper",
        "opt_name",
    }
    assert set(opt.meta.keys()) == param_keys.union(meta_keys)
