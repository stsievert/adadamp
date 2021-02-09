from copy import copy, deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, NewType

import dask
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from time import time
from distributed import get_client
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_random_state
from skorch import NeuralNet
from torch.autograd import Variable
from torch.utils.data import Dataset, IterableDataset, TensorDataset
from torch.nn.modules.loss import _Loss as Loss

IntArray = Union[List[int], np.ndarray, torch.Tensor]
Number = Union[int, float, np.integer, np.float]
Model = NewType("Model", torch.nn.Module)
Grads = NewType("Grads", Dict[str, Union[torch.Tensor, float, int]])
ArrayLike = Union[np.ndarray, torch.Tensor, List[Number]]


def gradient(
    model_opt: Tuple[Model, Optimizer],
    train_set,
    *,
    loss: Callable,
    device=torch.device("cpu"),
    idx: IntArray,
) -> Grads:
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
    model, _ = model_opt
    start = time()
    data_target = [train_set[i] for i in idx]
    _inputs = [d[0].reshape(-1, *d[0].size()) for d in data_target]
    _targets = [d[1] for d in data_target]
    inputs = torch.cat(_inputs)
    targets = torch.tensor(_targets)

    assert model.training, "Model should be in training model"
    outputs = model(inputs)

    _loss = loss(outputs, targets)
    _loss.backward()
    grads: Dict[str, torch.Tensor] = {k: v.grad for k, v in model.named_parameters()}
    elapsed = time() - start
    return {
        "_num_data": len(outputs),
        "_time": elapsed,
        "_loss": float(_loss.item()),
        **grads,
    }


def _update_model(
    model_opt: Tuple[Model, Optimizer], grads: List[Grads],
) -> Tuple[Model, Optimizer]:
    model, optimizer = model_opt

    # The deepcopy lines might be necessary -- see [1].
    # [1]:https://stackoverflow.com/questions/65257792/returning-mutated-inputs-with-dask
    #
    #  model = deepcopy(model)  # necessary?
    #  optimizer = deepcopy(optimizer)  # necessary?

    num_data = sum(info["_num_data"] for info in grads)

    # aggregate and update the gradients
    for name, param in model.named_parameters():
        grad = sum(g[name] for g in grads)
        param.grad = grad / num_data

    # update model
    optimizer.step()
    optimizer.zero_grad()
    return model, optimizer


class DaskBaseDamper:
    """
    Train a PyTorch model with Dask.

    This class has a Scikit-learn API. It accepts PyTorch datasets and
    array-like objects (PyTorch Tensors or NumPy arrays).

    Parameters
    ----------
    module : torch.nn.Module
    loss : PyTorch Loss
    optimizer : torch.optim.Optimizer
    batch_size : int
        The batch size to use.
    min_workers : int
        The minimum number of workers available
    max_workers : int
        The maximum number of workers available
    device : str
        The device used. Directly passed to :class:`torch.device`.
    grads_per_worker: int
        How many gradients should each worker handle?
    max_epochs : int
        The maximum number of passes through the dataset (or epochs) to train
        for.
    **kwargs : Dict[str, Any]
        Keyword arguments to pass to Skorch. For example, these can include
        ``optimizer__lr`` to set the learning rate or ``module__foo`` to pass
        keyword argument ``foo`` to module initialization.

    Attributes
    ----------
    module_ : nn.Module
        The initialize PyTorch module.
    optimizer_ : Optimizer
        The initialize PyTorch optimizer.
    loss_ : Loss
        The initialized PyTorch loss.
    initialized_ : bool
        Whether this class has been initialized.
    meta_ : Dict[str, Number]
        Meta information about the fitting process. This includes keys
        ``['n_updates', 'n_data', 'score__calls', 'partial_fit__calls']``.
    """
    def __init__(
        self,
        module: nn.Module,
        loss: Loss,
        optimizer: Optimizer,
        *,
        batch_size: int=32,
        min_workers: int=1,
        max_workers: int=8,
        device: str = "cpu",
        grads_per_worker: int=128,
        max_epochs: int=20,
        **kwargs,
    ):
        self.module = module
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.grads_per_worker = grads_per_worker
        self.max_epochs = max_epochs

        self.batch_size = batch_size
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

    def _initialize(self, *args, **kwargs):
        """Initialize the model/optimizer/etc"""
        module_kwargs = self._get_kwargs_for("module")
        opt_kwargs = self._get_kwargs_for("optimizer")
        loss_kwargs = self._get_kwargs_for("loss")

        self.module_ = self.module(**module_kwargs)
        self.optimizer_ = self.optimizer(self.module_.parameters(), **opt_kwargs)
        self.loss_ = self.loss(reduction="sum", **loss_kwargs)
        self._meta: Dict[str, Number] = {
            "n_updates": 0,
            "n_data": 0,
            "score__calls": 0,
            "partial_fit__calls": 0,
        }
        self._initialized = True

    def batch_size_(self):
        """The batch size, which is strongly related the noise in the
        gradient estimate. """
        return self.batch_size

    def _train_step(
        self,
        model_opt,
        dataset,
        *,
        loss,
        client,
        epoch_n_data=0,
        len_dataset=None,
        **fit_params,
    ):
        """
        Calculate gradients and take one optimization step with Dask.

        Parameters
        ----------
        X, y : torch.Tensors
            Input to model
        fit_params : dict
            Extra arguments passed to ``self.module_.forward``.

        Returns
        -------
        n_data : int
            The number of data processed.
        """
        bs = self.batch_size_()
        self.n_workers_ = bs // self.grads_per_worker
        if client.cluster:
            n_workers = np.clip(self.n_workers_, self.min_workers, self.max_workers)
            client.cluster.scale(int(n_workers))

        # compute grads
        grads = self._get_gradients(
            epoch_n_data,
            model_opt,
            dataset,
            batch_size=bs,
            loss=loss,
            client=client,
            n_workers=self.n_workers_,
            len_dataset=len_dataset,
        )
        model_opt = client.submit(_update_model, model_opt, grads)
        return model_opt, bs

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

    def fit(self, X: Union[ArrayLike, Dataset], y: ArrayLike=None, **fit_params):
        """
        Fit the model.

        Parameters
        ----------
        X : ArrayLike or Dataset
            Features if array-like, otherwise a torch dataset
        y : ArrayLike
        """
        self._initialize()
        for epoch in range(self.max_epochs):
            self.partial_fit(X, y=y, **fit_params)
        return self

    def partial_fit(self, X, y=None, **fit_params):
        """
        Runs 1 epoch on the given data and model

        Parameters
        ----------
        X : ArrayLike or Dataset
            Features if array-like, otherwise a torch dataset
        y : ArrayLike
        """
        start = time()
        if not self.initialized_:
            self._initialize()

        self._run_single_epoch(X, y=y, **fit_params)

        stat = {
            "partial_fit__time": time() - start,
            "partial_fit__batch_size": self.batch_size_(),
        }
        self._meta.update(stat)
        self._meta["partial_fit__calls"] += 1
        return self

    def _run_single_epoch(self, X, y=None, **fit_params):
        """
        Train a single epoch.

        X : np.ndarray, torch.Tensor, IterableDataset
            Inputs features or dataset.
        y : np.ndarray, torch.Tensor, optional.
            Outputs to match.
        fit_params : dict
            Arguments to pass to ``self.module_.forward``.
        """
        dataset = self._get_dataset(X, y=y)
        client = get_client()

        # Send the model/optimizer to workers
        m = deepcopy(self.module_.train())
        m = client.scatter(m)
        o = deepcopy(self.optimizer_)
        o = client.scatter(o)
        model_opt = client.submit(lambda x, y: (x, y), m, o)

        # Give all data to each worker
        len_dataset = len(dataset)
        dataset = client.scatter(dataset, broadcast=True)

        start_data = copy(self._meta["n_data"])
        loss = deepcopy(self.loss_)
        while True:
            model_opt, bs = self._train_step(
                model_opt,
                dataset,
                loss=loss,
                client=client,
                len_dataset=len_dataset,
                **fit_params,
            )
            self._meta["n_updates"] += 1
            self._meta["n_data"] += bs
            if self._meta["n_data"] - start_data >= len(X):
                break
        model, opt = model_opt.result()
        self.module_ = model
        self.optimizer_ = opt
        return True

    def score(self, X, y):
        dataset = self._get_dataset(X, y=y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1000)
        with torch.no_grad():
            _loss = 0
            for Xi, yi in loader:
                Xi, yi = Xi.to(self.device), yi.to(self.device)
                y_hat = self.module_.forward(Xi)
                _loss += self.loss_(y_hat, yi).item()
        return _loss / len(y)

    def _get_gradients(
        self,
        start_idx,
        model_opt,
        dataset,
        *,
        loss,
        batch_size,
        client,
        n_workers,
        len_dataset,
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

        """
        # Iterate through the dataset in batches
        # TODO: integrate with IterableDataset (this is pretty much already
        # an IterableDataset but without vectorization)
        idx = [i % len_dataset for i in range(start_idx, start_idx + batch_size)]
        worker_idxs = np.array_split(idx, n_workers)

        # Distribute the computation of the gradient. In practice this will
        # mean (say) 4 GPUs to accelerate the gradient computation. Right now
        # for ease it's a small network that doesn't need much acceleration.
        grads = [
            client.submit(gradient, model_opt, dataset, loss=loss, idx=idx)
            for idx in worker_idxs
        ]
        return grads

    @property
    def meta_(self):
        return deepcopy(self._meta)

    @property
    def initialized_(self):
        return hasattr(self, "_initialized") and self._initialized

    def _get_tags(self):
        return BaseEstimator()._get_tags()


class DaskClassifier(DaskBaseDamper):
    def score(self, X, y=None):
        """
        Score the current model on the data passed.

        Parameters
        ----------
        X : ArrayLike or Dataset
            Features if array-like, or a PyTorch Dataset.
        y : ArrayLike
            Targets if ``X`` is array-like.

        Raises
        ------
        NotFittedError
            If the estimator has not been fit yet
        ValueError
            If an empty array/dataset is passed.

        Notes
        -----
        The loss, time and accuracy are recorded in ``self.meta_``.
        """
        if not hasattr(self, "initialized_") or not self.initialized_:
            raise NotFittedError(
                "This instance is not fitted yet. Call 'fit' with the "
                "appropriate arguments before using this estimator"
             )

        if len(X) == 0:
            raise ValueError("Pass some data to `score`")

        if isinstance(X, torch.utils.data.Dataset):
            dataset = X
            loader = torch.utils.data.DataLoader(dataset, batch_size=1000)
        elif isinstance(X, torch.utils.data.DataLoader):
            loader = X
        else:
            dataset = self._get_dataset(X, y=y)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1000)

        correct = 0
        total = 0
        _loss = 0
        start = time()

        with torch.no_grad():
            for Xi, yi in loader:
                # loss
                Xi, yi = Xi.to(self.device), yi.to(self.device)
                y_hat = self.module_.forward(Xi)
                _loss += self.loss_(y_hat, yi).item()

                # acc
                _, index = y_hat.max(1)
                truth = (index == yi).long()
                correct += truth.sum()
                total += truth.shape[0]

        acc = float(correct / total)
        loss = float(_loss / total)

        # update stats
        stat = {
            "score__acc": acc,
            "score__loss": loss,
            "score__time": time() - start,
        }
        self._meta.update(stat)
        self._meta["score__calls"] += 1
        return acc
