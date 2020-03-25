import torch.utils.data
from torchvision import datasets, transforms
import torch.nn.functional as F
import pytest
import toolz
import numpy as np

from test_mnist import Net
from adadamp._dist import gradient


@pytest.fixture
def dataset():
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
    return (train_set, test_set)


@pytest.fixture
def model():
    return Net()


def test_basic_dist(dataset, model):
    model_params = sum(v.nelement() for v in model.parameters())
    train, test = dataset
    loss = F.nll_loss
    data_loader = torch.utils.data.DataLoader(train, batch_size=99)

    _grads = (
        gradient(input, output, model=model, loss=loss) for input, output in data_loader
    )
    grads = toolz.merge_with(sum, _grads)

    n_data = grads.pop("_num_data")
    assert n_data == 1000 == len(train)

    _grads = [v for v in grads.values() if isinstance(v, torch.Tensor)]
    grad_params = sum(v.nelement() for v in _grads)
    assert grad_params == model_params
    flat_grad = torch.cat([v.view(-1) for v in _grads])
    assert not np.isinf(flat_grad.abs().max().numpy())
