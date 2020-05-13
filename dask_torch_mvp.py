from dist_example_model import Net
from torchvision import datasets, transforms
from torchvision.transforms import Compose
from torchvision.datasets import FashionMNIST
from copy import copy
from distributed import Client, LocalCluster
import time

def _get_fashionmnist():
    """
    Gets FashionMINWT test and train data
    """
    transform_train = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ]
    transform_test = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    _dir = "_traindata/fashionmnist/"
    train_set = FashionMNIST(
        _dir, train=True, transform=Compose(transform_train), download=True,
    )
    test_set = FashionMNIST(_dir, train=False, transform=Compose(transform_test))
    return train_set, test_set


def train_model(model, train_set):
    # create client and scatter data
    cluster = LocalCluster(n_workers=8)
    client = Client(cluster)

    # I found the error! This ruins it! Only this line, or the one on 42 can run at once
    train_set_future = client.scatter(train_set, broadcast=True)

    for model_updates in range(5):

        # use the model to get the next grad step
        new_model = copy(model)
        new_model.eval()

        print("2 "*30)
        model_future = client.scatter(new_model)
        # ^^^^^ this line is the lat to get executred 
        print("BREAK "*999)

if __name__ == "__main__":
    # from to-joe
    kwargs = {
        "lr":0.0433062924,
        "batch_growth_rate": 0.3486433523,
        "dwell": 100,
        "max_batch_size": 1024,
        "grads_per_worker": 16,
        "initial_batch_size": 24,
        "initial_workers": 8,
        "epochs": 20_000,
    }
    model = Net()
    train_set, test_set = _get_fashionmnist()
    train_model(model, train_set)
