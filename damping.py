from typing import Callable, Union, Dict, Any, Tuple, Set
from pprint import pprint
from copy import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import BatchSampler, RandomSampler
from torch.optim import Optimizer
import torch.nn.functional as F
import torch.nn as nn


class BaseDamper(Optimizer):
    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        opt: Optimizer,
        loss: Callable = F.nll_loss,
        initial_batch_size: int = 1,
        device: str = "cpu",
        max_batch_size: Union[int, None] = None,
        reduction="mean",
        **kwargs,
    ):
        """
        Damp the noise in the gradient estimate.

        Arguments
        ---------
        model : nn.Module
            The model to train
        dataset : torch.Dataset
            Dataset to use for training
        opt : torch.optim.Optimizer
            The optimizer to use
        loss : callable (function), default=torch.nn.F.nll_loss
            The loss function to use
        initial_batch_size : int, default=1
            Initial batch size
        device : str, default="cpu"
            The device to use.
        max_batch_size : int, None, default=None
            The maximum batch size. If the batch size is larger than this
            value, the learning rate is decayed by an appropriate amount.
        reduction : str
            The loss reduction. By default, torch.nn.functional uses ``"mean"``
        kwargs : dict
            Arguments to pass to the underlying torch.DataLoader

        Notes
        -----
        By default, this class does not perform any damping (but it's children
        do). If a function needs an instance of BaseDamper, this class can wrap
        any optimizer.

        """
        self._params: Set[str] = {
            "device_type",
            "initial_batch_size",
            "loss_name",
            "max_batch_size",
            "reduction",
        }
        self.device = device
        self.initial_batch_size = initial_batch_size
        self.loss = loss
        self.max_batch_size = max_batch_size
        self.model = model
        self.reduction = reduction

        self._meta: Dict[str, Any] = {
            "model_updates": 0,
            "num_examples": 0,
            "batch_loss": None,
            "num_params": sum([m.nelement() for m in model.parameters()]),
            "len_dataset": len(dataset),
            "opt_params": opt.defaults,
        }

        self.opt = opt
        self.dataset = dataset
        self.loss = loss
        self.param_groups = self.opt.param_groups
        self.device = torch.device(device)
        sampler = RandomSampler(dataset, replacement=True)
        self.loader = DataLoader(dataset, sampler=sampler, drop_last=True, **kwargs,)

    def step(self, **kwargs):
        damping = self.damping()
        self.loader.batch_sampler.batch_size = int(
            self.initial_batch_size * int(damping)
        )

        # Is the batch size too large? If so, decay the learning rate
        current_bs = self.loader.batch_sampler.batch_size
        max_bs = self.max_batch_size
        if max_bs is not None and current_bs >= max_bs:
            self._set_lr(factor=max_bs / current_bs)
            self.sampler.batch_size = max_bs

        batch_loss, num_examples = self._step(**kwargs)

        self._meta["model_updates"] += 1
        self._meta["damping"] = damping
        self._meta["lr_"] = self._get_lr()
        self._meta["num_examples"] += num_examples
        self._meta["batch_loss"] = batch_loss
        self._meta["damping"] = damping
        self._meta["batch_size"] = self.loader.batch_sampler.batch_size

    def damping(self):
        return 1

    def _step(self, **kwargs):
        data, target = next(iter(self.loader))
        data, target = data.to(self.device), target.to(self.device)
        self.opt.zero_grad()
        output = self.model(data)
        loss = self.loss(output, target)
        loss.backward()
        self.opt.step(**kwargs)
        factor = 1 if self.reduction == "mean" else 1 / len(data)
        return loss.item() * factor, len(data)

    def _set_lr(self, factor=1):
        for group in self.opt.param_groups:
            group["lr"] *= factor
        return self.opt

    def _get_lr(self):
        lrs = [group["lr"] for group in self.opt.param_groups]
        assert all(lr == lrs[0] for lr in lrs)
        return lrs[0]

    def get_params(self):
        params = {k: v for k, v in self.__dict__.items() if k in self._params}
        return params

    @property
    def meta(self):
        d = copy(self._meta)
        d.update(self.get_params())
        d["device_type"] = self.device.type
        d["loss_name"] = self.loss.__name__
        d["epochs"] = d["num_examples"] / d["len_dataset"]
        return d

    def get_loss(
        self, dataset: Union[None, Dataset] = None, frac: Union[float, None] = None,
    ):
        if dataset is None:
            dataset = self.dataset
        num_eg = len(dataset)
        if frac is not None:
            num_eg = int(frac * len(dataset))
        if self.reduction not in ["mean", "sum"]:
            raise ValueError(
                "``reduction`` mis-specified (and is not 'mean' or 'sum')."
            )

        _sampler = RandomSampler(dataset, replacement=False)
        bs = np.clip(0.1 * len(dataset), 128 if self.device.type == "cuda" else 4, 1000)
        bs = int(bs)

        loader = DataLoader(dataset, sampler=_sampler, batch_size=bs)

        total_loss = 0
        _num_eg = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                _num_eg += len(data)
                factor = 1 if self.reduction == "sum" else len(data)
                output = self.model(data)
                loss = self.loss(output, target)
                total_loss += factor * loss.item()
                if _num_eg >= num_eg:
                    break

        return total_loss / _num_eg


class AdaDamp(BaseDamper):
    def damping(self):
        loss = self.get_loss()
        self._meta["_complete_loss"] = loss
        return _ceil(self.initial_batch_size / loss)


class PadaDamp(BaseDamper):
    def __init__(self, *args, rate=None, **kwargs):
        """
        Parameters
        ----------
        args : list
            Passed to BaseDamper

        rate : float
            The rate to increase the damping by. That is, set the batch size to be

            .. math::

                B_0 + ceil(rate * k)

            where k is the number of model updates.

        Notes
        -----
        The number of epochs is

        .. math::

            uB_0 + \sum_{i=1}^u ceil(rate * k)

        for u model updates.

        """
        self.rate = rate
        super().__init__(*args, **kwargs)

    def damping(self):
        k = self.meta["model_updates"]
        return self.initial_batch_size + _ceil(self.rate * k)


class GeoDamp(BaseDamper):
    def __init__(self, *args, geodelay=5, geofactor=2, **kwargs):
        self.geodelay = geodelay
        self.geofactor = geofactor
        super().__init__(*args, **kwargs)

    def damping(self):
        epochs = self.meta["num_examples"] / self.meta["len_dataset"]
        return self.geofactor ** ((epochs) // self.geodelay)


def train(
    model: nn.Module,
    opt: BaseDamper,
    verbose: Union[int, bool, None] = None,
    epochs=1,
    return_data: bool = False,
):
    """
    Function to train for at least one epoch.

    Arguments
    ---------
    model : nn.Module
        PyTorch model.
    opt : Union[AdaDamp, PadaDamp]
        Optimizer. Must be a subclass of BaseDamper
    verbose : int, float, None, default=None
        Controls printing. Higher values print more frequently, specifically
        approximately every ``1 / verbose`` fraction of the dataset;
        setting ``verbose == 10`` will mean it prints 10 times per epoch.

    Returns
    -------
    model : nn.Module
        The update model.

    """
    if not isinstance(opt, BaseDamper):
        raise ValueError(
            "Argument ``opt`` is not an instance of BaseDamper. "
            "(passing AdaDamp, PadaDamp or GeoDamp will resolve this issue)"
        )
    if verbose is not None:
        print_eg = int(len(opt.dataset) / verbose)
    start_examples = opt._meta["num_examples"]
    old_examples = opt._meta["num_examples"]
    data = []
    while True:
        if opt._meta["num_examples"] - start_examples >= len(opt.dataset):
            d = opt._meta["num_examples"] - start_examples
            break
        opt.step()
        if return_data:
            data.append(opt.meta)
        if verbose is not None and opt._meta["num_examples"] >= old_examples + print_eg:
            pprint(opt._meta)
            old_examples = opt._meta["num_examples"]
    if return_data:
        return model, opt, data
    return model, opt


def _ceil(x):
    return int(x) + 1
