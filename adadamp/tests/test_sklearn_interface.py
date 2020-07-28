import numpy as np
import pytest
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import sklearn.utils.estimator_checks as est_checks
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.datasets import make_regression
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose

from adadamp import DaskBaseDamper


class Net(nn.Module):
    def __init__(self, n_features=20, n_outputs=1, hidden=40):
        super(Net, self).__init__()

        self.hidden = hidden
        self.n_features = n_features
        self.n_outputs = n_outputs

        self.fc1 = nn.Linear(self.n_features, self.hidden)
        self.fc2 = nn.Linear(self.hidden, self.n_outputs)

    def forward(self, x):
        x = x.float()  # type casting to make sure works well with sklearn
        return F.log_softmax(self.fc2(F.relu(self.fc1(x))), dim=1)


@pytest.fixture
def X():
    X, y = make_regression(random_state=42, n_features=20)
    return torch.from_numpy(X.astype("float32"))


@pytest.fixture
def y():
    X, y = make_regression(random_state=42, n_features=20)
    return torch.from_numpy(y.astype("float32"))


def test_pytorch_dataset(X, y):
    dataset = torch.utils.data.TensorDataset(X, y)

    net = DaskBaseDamper(
        module=Net, loss=nn.MSELoss, optimizer=optim.SGD, optimizer__lr=0.05
    )
    net.get_params()
    net.fit(dataset)


def test_numpy_data(X, y):
    X = X.numpy()
    y = y.numpy()

    net = DaskBaseDamper(
        module=Net, loss=nn.MSELoss, optimizer=optim.SGD, optimizer__lr=0.05
    )
    net.get_params()
    net.fit(X, y)

from sklearn.utils.estimator_checks import check_estimator


class SklearnComptabilityNet(nn.Module):
    def __init__(self, n_features=20, n_outputs=1, hidden=40):
        super().__init__()
        #  self.coef = torch.randn(1, dtype=torch.float32, requires_grad=True)
        self.coef = nn.Linear(1, 1)

    def forward(self, x):
        # type casting to make sure works well with sklearn
        first_feature = x[:, 0].resize(len(x), 1).float()
        return self.coef(first_feature)

class SklearnComptabilityLoss(nn.MSELoss):
    def forward(self, inputs, targets):
        return super().forward(inputs.float(), targets.float())

est_checks = list(
    check_estimator(
        DaskBaseDamper(
            module=SklearnComptabilityNet,
            loss=SklearnComptabilityLoss,
            optimizer=optim.SGD,
            optimizer__lr=0.05,
        ),
        generate_only=True,
    )
)

@pytest.mark.parametrize(
    "estimator,check", est_checks, ids=[check.func.__name__ for _, check in est_checks]
)
def test_sklearn_compatible_estimator(estimator, check, request):
    better_module = "handle this in custum module; up to the user"
    conv_msg = "PyTorch doesn't support converting {} arrays"
    reasons = {
        "no_attributes_set_in_init": "dirty hack",
        "fit1d": better_module,
        "nan_inf": better_module,
        "complex_data": conv_msg.format("complex"),
        "dtype_object": conv_msg.format("object"),
        "sparse": conv_msg.format("sparse"),
        "empty": "Empty list passed",
    }

    name = check.func.__name__
    if any(n in name for n in reasons):
        shorthand = [n for n in reasons if n in name]
        assert len(shorthand) == 1
        reason = reasons[shorthand[0]]
        pytest.xfail(reason=reason)
    check(estimator)
