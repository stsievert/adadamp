from __future__ import print_function
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from adadamp import PadaDamp


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


def test_main():
    from adadamp.experiment import train, test

    # Training settings
    args = SimpleNamespace(
        batch_size=1024,
        epochs=2,
        log_interval=10,
        lr=0.1,
        no_cuda=False,
        save_model=False,
        seed=1,
        test_batch_size=1000,
    )

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_set = datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    test_set = datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )

    # Only for making tests run faster
    dataset, _ = torch.utils.data.random_split(train_set, [2000, len(train_set) - 2000])
    train_set, test_set = torch.utils.data.random_split(dataset, [1000, 1000])
    test_loader = DataLoader(test_set, batch_size=300)

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    model = Net().to(device)
    _optimizer = optim.SGD(model.parameters(), lr=args.lr)
    loss = F.nll_loss
    optimizer = PadaDamp(
        model=model,
        dataset=train_set,
        opt=_optimizer,
        loss=loss,
        device="cpu",
        batch_growth_rate=0.1,
        initial_batch_size=32,
        max_batch_size=1024,
    )

    print("Starting...")
    for epoch in range(1, args.epochs + 1):
        train(model=model, opt=optimizer)
        data = test(model=model, loss=loss, loader=test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
