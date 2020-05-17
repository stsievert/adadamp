from torchvision.datasets import FashionMNIST
from copy import deepcopy
from distributed import Client
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Example custom model
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 5, stride=1)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    # setup model and net
    client = Client()
    #  model = resnet18()  # code passes if uncommented
    model = Net()
    # get data
    train_set = FashionMNIST("_traindata/fashionmnist/", train=True, download=True)
    # set model to dask, ensure error free
    model_future = client.scatter(deepcopy(model))
    assert True, "Sanity check to make sure reached"
    # Hm... this is where our bug occurs, when scatter the model following our trian set
    ts_f3 = client.scatter(train_set, broadcast=True)
    model_future = client.scatter(deepcopy(model))

