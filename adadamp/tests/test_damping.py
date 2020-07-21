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
from torch.utils.data import TensorDataset, random_split

from adadamp import BaseDamper, GeoDamp, AdaDamp, PadaDamp, GradientDescent
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
def images():
    d = datasets.MNIST(
        "_traindata/mnist/",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    n = 1000
    d, _ = random_split(d, [n, len(d) - n])
    return d


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 16, 3, 1)
        self.fc1 = nn.Linear(16 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


@pytest.fixture
def large_dataset():
    X, y = make_classification(
        n_features=20, n_samples=3000, n_classes=10, n_informative=10
    )
    return TensorDataset(
        torch.from_numpy(X.astype("float32")), torch.from_numpy(y.astype("int64"))
    )


@pytest.fixture
def model():
    return Net().to("cpu")


def test_basics(model, dataset, epochs=14):
    optimizer = optim.Adadelta(model.parameters(), lr=1)
    opt = BaseDamper(model, dataset, optimizer, initial_batch_size=8, dwell=1)

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


def test_basics_imgs(images):
    model = MNISTNet()
    test_basics(model, images, epochs=2)


def test_geodamp(model, dataset):
    _opt = optim.Adadelta(model.parameters(), lr=1)
    opt = GeoDamp(
        model,
        dataset,
        _opt,
        initial_batch_size=1,
        dampingdelay=4,
        dampingfactor=2,
        dwell=1,
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


def test_large_batch_size(model, large_dataset):
    _opt = optim.Adadelta(model.parameters(), lr=1)
    opt = BaseDamper(model, large_dataset, _opt, initial_batch_size=1024, dwell=1)
    data: List[Dict[str, Any]] = []
    data2: List[Dict[str, Any]] = []
    for epoch in range(1, 16 + 1):
        model, opt, meta, _ = experiment.train(model, opt)
        data.append(opt.meta)
        data2.append(meta)
    df = pd.DataFrame(data)

    # Make sure the loss is decreasing
    assert df.batch_loss.diff().median() < -0.01
    assert df.batch_loss.diff().mean() < -0.01
    assert 2.25 < df.loc[0, "batch_loss"]
    assert df.loc[15, "batch_loss"] < 2.06


def test_padadamp(model, dataset):
    _opt = optim.Adadelta(model.parameters(), lr=1)
    opt = PadaDamp(
        model, dataset, _opt, batch_growth_rate=1, initial_batch_size=4, dwell=1
    )
    data: List[Dict[str, Any]] = []
    for epoch in range(1, 16 + 1):
        model, opt, meta, train_data = experiment.train(model, opt)
        data += train_data
    df = pd.DataFrame(data)
    assert (df.damping >= 1).all()
    # Commented out because PadaDamp uses exponential growth
    #  assert (df.damping == np.ceil(df.model_updates)).all()


def test_adadamp(model, dataset):
    init_bs = 8
    _opt = optim.SGD(model.parameters(), lr=0.500)
    opt = AdaDamp(
        model, dataset, _opt, initial_batch_size=init_bs, dwell=1, approx_loss=False,
    )
    data: List[Dict[str, Any]] = []
    initial_loss = opt._get_loss()
    for epoch in range(5):
        model, opt, meta, train_data = experiment.train(model, opt)
        data += train_data
    df = pd.DataFrame(data)

    bs_hat = init_bs * df.loc[0, "_initial_loss"] / df._last_loss
    bs_hat = np.ceil(bs_hat.values).astype(int)
    bs = df.batch_size.values
    assert (bs_hat <= bs).all()
    #  assert (bs == bs_hat).all()


def test_gradient_descent(model, dataset):
    init_bs = 8
    _opt = optim.SGD(model.parameters(), lr=0.500)
    opt = GradientDescent(model, dataset, _opt, dwell=1)
    data: List[Dict[str, Any]] = []
    initial_loss = opt._get_loss()
    for epoch in range(5):
        model, opt, meta, train_data = experiment.train(model, opt)
        data += train_data
    df = pd.DataFrame(data)
    assert (df.batch_loss.diff().dropna() < 0).all()
    assert (df.len_dataset == df.batch_size).all()
    assert np.allclose(df.epochs.diff().dropna(), 1)


def test_avg_loss(model, dataset):
    """
    Test that BaseDamper._get_loss returns mean loss regardless of how many
    points are sampled.
    """
    _opt = optim.Adadelta(model.parameters(), lr=1)
    opt = BaseDamper(model, dataset, _opt, dwell=1, initial_batch_size=32)
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
    opt = BaseDamper(model, dataset, _opt, dwell=1)
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
        "dwell",
        "random_state",
    }
    assert set(opt.meta.keys()) == param_keys.union(meta_keys)


@pytest.mark.parametrize("dwell", [1, 2, 4, 8, 16])
def test_dwell(dwell, model, dataset):
    _opt = optim.Adadelta(model.parameters(), lr=1)
    # batch_growth_rate=1: every model update increase the batch size by 1
    opt = PadaDamp(
        model, dataset, _opt, dwell=dwell, initial_batch_size=4, batch_growth_rate=1
    )
    data = []
    for epoch in range(1, 16 + 1):
        model, opt, meta, train_data = experiment.train(model, opt)
        data.extend(train_data)
    df = pd.DataFrame(data)

    # Because geometric delay... (tested below)
    damping = df.damping.iloc[dwell:]

    chunks = [
        damping.iloc[dwell * k : dwell * (k + 1)].values
        for k in range(len(df) // dwell)
    ]
    chunks = [c for c in chunks if len(c)]
    if dwell > 1:
        assert all(np.allclose(np.diff(c), 0) for c in chunks[1:])
    else:
        assert all(len(c) <= 1 for c in chunks)


def test_dwell_init_geo_increase(model, dataset):
    dwell = 512
    _opt = optim.Adagrad(model.parameters(), lr=1)
    # batch_growth_rate=1: every model update increase the batch size by 1
    opt = PadaDamp(
        model, dataset, _opt, dwell=dwell, initial_batch_size=4, batch_growth_rate=1
    )
    data = []
    for epoch in range(1, 16 + 1):
        model, opt, meta, train_data = experiment.train(model, opt)
        data.extend(train_data)
    df = pd.DataFrame(data)
    cbs = np.arange(64) + 1  # cnts batch size
    dbs = [[cbs[2 ** i]] * 2 ** i for i in range(4)]  # discrete bs
    dbs = sum(dbs, [])
    assert len(dbs) == 15
    assert (np.array(dbs) == df.batch_size.iloc[1 : 1 + len(dbs)]).all()


def test_lr_decays(model, dataset):
    _opt = optim.SGD(model.parameters(), lr=1)
    # batch_growth_rate=1: every model update increase the batch size by 1
    opt = GeoDamp(
        model,
        dataset,
        _opt,
        dwell=1,
        initial_batch_size=4,
        dampingdelay=3,
        dampingfactor=2,
        max_batch_size=8,
    )
    data = []
    for epoch in range(1, 16 + 1):
        model, opt, meta, train_data = experiment.train(model, opt)
        data.extend(train_data)
    df = pd.DataFrame(data)
    damping_factor = df.damping / 4

    # Damping always increasing/decreasing
    assert (np.diff(df.batch_size) >= 0).all()
    assert (np.diff(df.lr_) <= 0).all()
    assert (np.diff(damping_factor) >= 0).all()

    # Make sure increases by correct amounts
    assert set(damping_factor.unique()) == {1, 2, 4, 8, 16, 32}
    assert set(df.batch_size) == {4, 8}
    assert set(df.lr_) == {1, 1/2, 1/4, 1/8, 1/16}

