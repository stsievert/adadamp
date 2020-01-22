# modified from [1]
# [1]:https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import damping


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


if __name__ == "__main__":
    args = SimpleNamespace(
        batch_size=64,
        epochs=14,
        gamma=0.7,
        log_interval=10,
        lr=1.0,
        no_cuda=False,
        save_model=False,
        seed=1,
        test_batch_size=1000,
    )
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    normalize = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_set = datasets.MNIST("../data", train=True, transform=normalize)
    test_set = datasets.MNIST("../data", train=False, transform=normalize)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = damping.AdaDamp(model, train_set, optimizer)

    for epoch in range(1, args.epochs + 1):
        damping.train(model, optimizer, print_freq=10)
