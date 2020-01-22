# modified from https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
from types import SimpleNamespace
from typing import Any, Dict, List

from sklearn.datasets import make_classification
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset

import damping


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


if __name__ == "__main__":
    args = SimpleNamespace(
        initial_batch_size=16,
        epochs=14,
        log_interval=10,
        lr=1.0,
        no_cuda=False,
        seed=1,
        test_batch_size=1000,
    )
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    X, y = make_classification(
        n_features=20, n_samples=100, n_classes=10, n_informative=10
    )
    train_set = TensorDataset(
        torch.from_numpy(X.astype("float32")), torch.from_numpy(y.astype("int64"))
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    opt = damping.AdaDamp(
        model, train_set, optimizer, initial_batch_size=args.initial_batch_size
    )

    data: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model, opt = damping.train(model, opt, print_freq=None)
        data.append(opt.meta)

    df = pd.DataFrame(data)
    assert (df.model_updates * df.batch_size == df.num_examples).all()
    df["epochs"] = df.num_examples / df.len_dataset
    assert df.epochs.max() <= args.epochs + 2
    eg_per_epoch = df.num_examples.diff().iloc[1:]
    len_dataset = df.len_dataset.iloc[1:]
    assert all((eg_per_epoch - len_dataset) <= df.batch_size.iloc[1:])
