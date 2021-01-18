from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from pathlib import Path
from notebooks import model
from sklearn.datasets import make_classification

DIR = Path(".").absolute()
sys.path.append(str(DIR))
os.chdir(str(DIR.parent)) # make notebook assume its in parent dir

def get_model_weights(model):
    s = 0
    for param in model.parameters():
        s += torch.abs(torch.sum(param)).item()
    return s

def get_dataset():
    # Expected 4-dimensional input for 4-dimensional weight [32, 1, 3, 3], but got 2-dimensional input of size [32, 20] instead
    X, y = make_classification(n_samples=batch_size * n_updates, n_features=784)
    # match MNIST X
    X.resize((640, 28, 28))
    X = torch.from_numpy(X)
    X = X.unsqueeze(1)
    X = X.float()
    X.to(device)
    # match MNIST y
    y = torch.from_numpy(y)
    y.to(device)
    
    return X, y
    
    
if __name__ == "__main__":
    
    batch_size = 128
    n_updates = 5
    device_str = "cpu" if not torch.cuda.is_available() else "cuda:0"
    device = torch.device(device_str)
    kwargs = {
        'batch_size': batch_size, 
        'max_epochs': 1, 
        'random_state': 42,
        'module': Net,
        'weight_decay': 1e-5,
        'loss': nn.CrossEntropyLoss,
        'optimizer': optim.Adagrad,
        'device': device_str
    }
    
    X, y = get_dataset()

    # call fit with whole dataset
    est1 = DaskClassifier(**kwargs)
    est1.fit(X, y)
    print(get_model_weights(est1.module_))

    # run partial fit many times
    est2 = DaskClassifier(**kwargs)
    for k in range(n_updates):
        idx = np.arange(batch_size * k, batch_size * (k + 1)).astype(int)
        est2.partial_fit(X[idx], y[idx])
    print(get_model_weights(est2.module_))

    assert np.allclose(get_model_weights(est1.module_), get_model_weights(est2.module_))