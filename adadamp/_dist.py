from copy import copy, deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import dask
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from distributed import get_client
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from skorch import NeuralNet
from torch.autograd import Variable
from torch.utils.data import Dataset, IterableDataset, TensorDataset

IntArray = Union[List[int], np.ndarray, torch.Tensor]


def gradient(
    train_set,
    *,
    model: nn.Module,
    loss: Callable,
    device=torch.device("cpu"),
    idx: IntArray,
) -> Dict[str, Union[torch.Tensor, int]]:
    r"""
    Compute the model gradient for the function ``loss``.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to eval

    inputs : torch.Tensor
        Input array to model

    targets : torch.Tensor
        Target output for model. The gradient is computed with respect to this.

    device : torch.device

    loss : Callabale
        PyTorch loss. Should return the loss with

        .. code:: python

           loss(output, out_, reduction="sum")

    idx : Optional[IntArray], optional
        The indices to compute the gradient over.

    Returns
    -------
    grad : Dict[str, Union[Tensor, int]]
        Gradient. This dictionary has all the keys in
        ``model.named_parameters.keys()``.

    Notes
    -----
    This function computes the gradient of the *sum* of inputs, not the *mean*
    of inputs. Functionally, this means evaluates the gradient of
    :math:`\sum_{i=1}^B l(...)`, not :math:`\frac{1}{n} \sum_{i=1}^B l(...)`
    where `l` is the loss function for a single example.

    """
    data_target = [train_set[i] for i in idx]
    _inputs = [d[0].reshape(-1, *d[0].size()) for d in data_target]
    _targets = [d[1] for d in data_target]
    inputs = torch.cat(_inputs)
    targets = torch.tensor(_targets).reshape(-1, 1)

    model.train()
    outputs = model(inputs)

    _loss = loss(targets, outputs)
    _loss.backward()
    grads = {k: v.grad for k, v in model.named_parameters()}
    return {"_num_data": len(outputs), "_loss": _loss.item(), **grads}


class DaskBaseDamper:
    def __init__(
        self,
        module,
        loss,
        optimizer,
        *,
        metrics=None,
        batch_size=32,
        cluster=None,
        max_batch_size=1024,
        min_workers=1,
        max_workers=8,
        device: str = "cpu",
        grads_per_worker=32,
        **kwargs,
    ):
        self.module = module
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.device = device
        self.cluster = cluster
        self.grads_per_worker = grads_per_worker

        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.min_workers = min_workers
        self.max_workers = max_workers
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _get_param_names(self):
        return [k for k in self.__dict__ if k[0] != "_" and k[-1] != "_"]

    def get_params(self, deep=True, **kwargs):
        params = BaseEstimator.get_params(self, deep=deep, **kwargs)
        return params

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self

    def _get_kwargs_for(self, name):
        name = f"{name}__"
        return {
            k.replace(name, ""): v
            for k, v in self.get_params().items()
            if name in k[: len(name) + 4]
        }

    def initialize(self, *args, **kwargs):
        module_kwargs = self._get_kwargs_for("module")
        opt_kwargs = self._get_kwargs_for("optimizer")
        loss_kwargs = self._get_kwargs_for("loss")

        self.module_ = self.module(**module_kwargs)
        self.optimizer_ = self.optimizer(self.module_.parameters(), **opt_kwargs)
        self.loss_ = self.loss(reduction="sum", **loss_kwargs)
        self.meta_ = {"n_updates": 0, "n_data": 0}
        self.initialized_ = True

    def batch_size_(self):
        return 32

    def train_step(self, dataset, client=None, epoch_n_data=0, **fit_params):
        """
        Calculate gradients and take one optimization step.

        Parameters
        ----------
        X, y : torch.Tensors
            Input to model
        fit_params : dict
            Extra arguments passed to module_.forward.

        Returns
        -------
        n_data : int
            The number of data processed.
        """
        # Send the model to workers
        new_model = deepcopy(self.module_).train()
        if client:
            model_future = client.scatter(new_model, broadcast=True)
        else:
            model_future = new_model

        bs = self.batch_size_()
        self.n_workers_ = bs // self.grads_per_worker
        if self.cluster:
            self.cluster.scale(self.n_workers_)

        # compute grads
        self.optimizer_.zero_grad()
        grads = self._get_gradients(
            epoch_n_data,
            new_model,
            dataset,
            batch_size=bs,
            n_workers=self.n_workers_,
            client=client,
        )

        # update SGD
        num_data = sum(info["_num_data"] for info in grads)
        assert num_data == bs, "Sanity check on batch size"

        # aggregate and update the gradients
        #  print(grads)
        for name, param in self.module_.named_parameters():
            grad = sum(g[name] for g in grads)
            param.grad = grad / num_data

        self.optimizer_.step()
        self.optimizer_.zero_grad()
        return bs

    def _get_dataset(self, X, y=None) -> Dataset:
        if isinstance(X, Dataset):
            return X
        if isinstance(X, list):
            X = np.ndarray(X)
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if isinstance(y, list) and len(y):
            y = np.ndarray(y)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        args = (X, y) if y is not None else (X,)
        return TensorDataset(*args)

    def partial_fit(self, X, y=None, **fit_params):
        if not hasattr(self, "initialized_") or not self.initialized_:
            self.initialize()
        self.run_single_epoch(X, y=y, **fit_params)
        return self

    fit = partial_fit

    def run_single_epoch(self, X, y=None, **fit_params):
        """
        Train a single epoch.

        X : np.ndarray, torch.Tensor, IterableDataset
            Inputs features or dataset.
        y : np.ndarray, torch.Tensor, optional.
            Outputs to match.
        fit_params : dict
            Arguments to pass to self.module_.forward
        """
        dataset = self._get_dataset(X, y=y)
        start_data = copy(self.meta_["n_data"])
        while True:
            bs = self.train_step(dataset, **fit_params)
            self.meta_["n_updates"] += 1
            self.meta_["n_data"] += bs
            if self.meta_["n_data"] - start_data >= len(X):
                break
        return True

    def score(self, X, y):
        dataset = self._get_dataset(X, y=y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1000)
        with torch.no_grad():
            _loss = 0
            for Xi, yi in loader:
                Xi, yi = Xi.to(self.device), yi.to(self.device)
                y_hat = self.module_.forward(Xi)
                _loss += self.loss_(yi.reshape(-1, 1), y_hat).item()
        return _loss / len(y)

    def _get_gradients(
        self, start_idx, model_future, dataset, batch_size, n_workers, client=None,
    ):
        """
        Calculates the gradients at a given state. This function is
        mainly in charge of sending the gradient calculation off to the
        dask workers, the hard work is done in _dist.gradient()

        Note:
        - In this implementation, each worker manually grabs the data
          and caches it themselves. This is a work around for a dask bug
        - See the doc string of _get_fashionminst in _dist.py for more info

        Parameters
        ----------
        start_idx : int
            Index to start sampling at. Data indices from ``start_idx`` to
            ``start_idx + batch_size`` will be sampled.

        model_future : distributed.future (?)
            Future object of the model to get gradients for

        n : int
            number of items in training set

        idx: int
            Current epoch, used to set up random state

        batch_size: int
            Current batch size

        n_workers: int
            Number of workers current running

        client : distributed.Client()
            Dask client used to scheduele workers. Assumes client
            is backed by some cluster (ie LocalCluster or equiv)

        """
        # Iterate through the dataset in batches
        # TODO: integrate with IterableDataset (this is pretty much already
        # an IterableDataset but without vectorization)
        n = len(dataset)
        idx = [i % n for i in range(start_idx, start_idx + batch_size)]
        worker_idxs = np.array_split(idx, n_workers)

        # Distribute the computation of the gradient. In practice this will
        # mean (say) 4 GPUs to accelerate the gradient computation. Right now
        # for ease it's a small network that doesn't need much acceleration.
        grad_fn = partial(gradient, dataset, model=model_future, loss=self.loss_)
        grad_fn = dask.delayed(grad_fn)

        grads = [grad_fn(idx=idx) for idx in worker_idxs]
        out = dask.compute(*grads)
        return out
