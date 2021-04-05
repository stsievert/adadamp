import dask
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.optim import Optimizer
from time import sleep, time
from distributed import get_client
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from copy import copy, deepcopy
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, NewType
from warnings import warn
from torch.autograd import Variable
from torch.utils.data import Dataset, IterableDataset, TensorDataset, DataLoader

IntArray = Union[List[int], np.ndarray, torch.Tensor]
Number = Union[int, float, np.integer, np.float]
Model = NewType("Model", torch.nn.Module)
Grads = NewType("Grads", Dict[str, Union[torch.Tensor, float, int]])


def _get_model_weights(model: nn.Module) -> float:
    """
    Computes the sum of the model weights for the passed model
    """
    with torch.no_grad():
        s = 0.0
        params = model.parameters()
        for k, param in enumerate(params):
            s += torch.linalg.norm(param).item()
            if k >= 2:
                break
    return s


def _set_random_state(seed: int, device="cuda") -> bool:
    """
    Sets random state based on a seed
    """
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if "cuda" in device:
        torch.cuda.manual_seed_all(seed)


def gradient(
    model_opt: Tuple[Model, Optimizer],
    train_set,
    *,
    loss: Callable,
    device=torch.device("cpu"),
    idx: IntArray,
    max_bs: int = 1024,
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
    # Workaround: Gradients should be cleared when entering this funciton,
    #     but at this moment this behavior is not occuring
    model = deepcopy(model_opt[0])

    start = time()

    # set up data
    idx.sort()
    _data, _target = train_set[idx]

    # split by max bastch size
    Data = torch.split(_data, max_bs)
    Target = torch.split(_target, max_bs)

    # Zero gradients
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= 0.0

    if not model.training:
        model = model.train()

    # run through in batches tracking net result
    loss_agg = 0.0
    n_items = 0
    for data, target in zip(Data, Target):
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _loss = loss(outputs, target)
        _loss.backward()
        loss_agg += float(_loss.item())
        n_items += len(outputs)

    grads: Dict[str, torch.Tensor] = {k: v.grad for k, v in model.named_parameters()}
    elapsed = time() - start
    return {
        "_num_data": n_items,
        "_time": elapsed,
        "_loss": loss_agg,
        **grads,
    }


def _update_model(
    model_opt: Tuple[Model, Optimizer], grads: List[Grads]
) -> Tuple[Model, Optimizer]:

    model, optimizer = model_opt
    num_data = sum(info["_num_data"] for info in grads)

    old_weights = _get_model_weights(model)

    # aggregate and update the gradients
    for name, param in model.named_parameters():
        grad = sum(g[name] for g in grads)  # sums together all grads for this layer
        param.grad = grad / num_data

    # Make sure the model and optimizer are connected
    init_lr = None
    for g in optimizer.param_groups:
        init_lr = float(g["lr"])

    optimizer.param_groups = []
    optimizer.add_param_group({"params": list(model.parameters())})

    # Re-set learning rate from initial optimizer
    for g in optimizer.param_groups:
        g["lr"] = init_lr

    # update model
    optimizer.step()
    optimizer.zero_grad()

    new_weights = _get_model_weights(model)

    rel_error = abs(new_weights - old_weights) / abs(old_weights)
    if np.allclose(rel_error, 0):
        warn(f"Model appears not to update with diff={rel_error}")

    return model, optimizer


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
        max_batch_size=None,
        lr=1,
        worker_max_batch_size=1024,
        min_workers=1,
        max_workers=8,
        device: str = "cpu",
        grads_per_worker=32,
        max_epochs=20,
        random_state=None,
        **kwargs,
    ):
        self.module = module
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.device = device
        self.cluster = cluster
        self.grads_per_worker = grads_per_worker
        self.max_epochs = max_epochs
        self.lr = lr

        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.worker_max_batch_size = worker_max_batch_size
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.n_workers_ = min_workers
        self.random_state = random_state

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def preprocess(ds: Dataset) -> TensorDataset:
        """
        Turn a PyTorch dataset into Torch Tensors.

        This will require a moderate amount of memory. For image datasets where
        each pixel is transformed from uint8 to float32, this will require a
        factor of 4 more memory. For the popular MNIST and CIFAR datasets, this
        means 37MB → 150MB and 175MB → 703MB respectively.

        These data are not stored on the training devices (/GPUs), so
        essentially only the transformations are preformed.

        """
        loader = DataLoader(ds, shuffle=False, batch_size=1000)
        data_target = [d for d in loader]
        n_tensors = len(data_target[0])
        tensors = []
        for i in range(n_tensors):
            tensor = [d[i] for d in data_target]
            tensors.append(torch.cat(tensor))
        return TensorDataset(*tensors)

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
        if self.max_batch_size is not None and self.lr is None:
            raise ValueError("lr needs to be set if max_batch_size is set")
        module_kwargs = self._get_kwargs_for("module")
        opt_kwargs = {"lr": self.lr}
        opt_kwargs.update(self._get_kwargs_for("optimizer"))
        loss_kwargs = self._get_kwargs_for("loss")
        self.random_state_ = check_random_state(self.random_state)

        limit = 2 ** 32 - 1
        _set_random_state(self.random_state_.randint(limit))
        client = get_client()
        workers = client.scheduler_info()["workers"].keys()
        for worker in workers:
            client.run(_set_random_state, self.random_state_.randint(limit))

        self.module_ = self.module(**module_kwargs)
        self.module_.to(torch.device(self.device))
        self.optimizer_ = self.optimizer(self.module_.parameters(), **opt_kwargs)
        self.loss_ = self.loss(reduction="sum", **loss_kwargs)
        self.damping_ = self.batch_size
        self._meta: Dict[str, Number] = {
            "n_updates": 0,
            "n_data": 0,
            "score__calls": 0,
            "partial_fit__calls": 0,
            "n_workers": self.n_workers_,
        }
        self.initialized_ = True
        return self

    @property
    def damping_(self):
        return self._damping

    @damping_.setter
    def damping_(self, d: float):
        self._damping = d

    def train_step(
        self,
        model_opt,
        dataset,
        *,
        loss,
        client,
        epoch_n_data=0,
        len_dataset=None,
        device=None,
        **fit_params,
    ):
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
        damping = self.damping_
        lr = self._get_lr()
        bs = damping
        if self.max_batch_size and self.lr and bs > self.max_batch_size:
            factor = self.max_batch_size / bs
            assert factor < 1, factor
            lr = self.lr * factor
            self._set_lr(lr)
            bs = self.max_batch_size
        self._meta.update({"damping_": damping, "batch_size_": bs, "lr_": lr})

        self.n_workers_ = bs // self.grads_per_worker
        if self.cluster:
            self.cluster.scale(self.n_workers_)

        self._meta.update({"n_workers": self.n_workers_})

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
            device=device,
        )

        model_opt = client.submit(_update_model, model_opt, grads)

        return model_opt, bs

    def _get_dataset(self, X, y=None) -> TensorDataset:
        if isinstance(X, TensorDataset):
            return X
        elif isinstance(X, Dataset):
            return self.preprocess(X)
        else:
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

    def fit(self, X, y=None, **fit_params):
        for epoch in range(self.max_epochs):
            self.partial_fit(X, y=y, **fit_params)
        return self

    def partial_fit(self, X, y=None, **fit_params):
        start = time()
        if not hasattr(self, "initialized_") or not self.initialized_:
            self.initialize()

        self.run_single_epoch(X, y=y, **fit_params)
        self._meta["partial_fit__time"] = time() - start
        self._meta["partial_fit__calls"] += 1
        return self

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
        client = get_client()

        self.module_.to(self.device)

        # Send the model/optimizer to workers
        m = client.scatter(self.module_)

        o = client.scatter(self.optimizer_)
        model_opt = client.submit(lambda x, y: (x, y), m, o)

        # Give all data to each worker
        len_dataset = len(dataset)
        dataset = client.scatter(dataset, broadcast=True)

        # Track when we have moved through dataset
        start_data = copy(self._meta["n_data"])
        loss = deepcopy(self.loss_)
        device = torch.device(self.device)

        # Run BS items through until hitting every element in dataset
        while True:
            model_opt, bs = self.train_step(
                model_opt,
                dataset,
                loss=loss,
                client=client,
                len_dataset=len_dataset,
                device=device,
                **fit_params,
            )

            # exit condition
            self._meta["n_updates"] += 1
            self._meta["n_data"] += bs
            if self._meta["n_data"] - start_data >= len(X):
                break

        m, o = model_opt.result()
        self.module_ = m
        self.optimizer_ = o
        return True

    def score(self, X, y):
        dataset = self._get_dataset(X, y=y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1000)
        device = torch.device(self.device)
        with torch.no_grad():
            _loss = 0
            for Xi, yi in loader:
                Xi, yi = Xi.to(device), yi.to(device)
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
        device,
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
        idx = self.random_state_.choice(
            len_dataset, size=min(batch_size, len_dataset), replace=False
        )
        idx.sort()
        worker_idxs = np.array_split(idx, n_workers)

        # Distribute the computation of the gradient. In practice this will
        # mean (say) 4 GPUs to accelerate the gradient computation. Right now
        # for ease it's a small network that doesn't need much acceleration.
        grads = [
            client.submit(
                gradient,
                model_opt,
                dataset,
                device=device,
                loss=loss,
                idx=idx,
                max_bs=self.worker_max_batch_size,
            )
            for idx in worker_idxs
        ]
        return grads

    @property
    def meta_(self):
        return deepcopy(self._meta)

    def _get_tags(self):
        return BaseEstimator()._get_tags()

    def _get_lr(self) -> float:
        if not hasattr(self, "initialized_") or not self.initialized_:
            self.initialize()

        for g in self.optimizer_.param_groups:
            break
        return g["lr"]

    def _set_lr(self, lr: float):
        if not hasattr(self, "initialized_") or not self.initialized_:
            self.initialize()

        for g in self.optimizer_.param_groups:
            g["lr"] = lr


class DaskClassifier(DaskBaseDamper):
    def score(self, X, y=None):
        if not hasattr(self, "initialized_") or not self.initialized_:
            self.initialize()

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


