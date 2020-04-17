from typing import Callable, Dict, Any, Tuple, Set, List, Optional, Union
from pprint import pprint
from copy import copy
import itertools
from time import time

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torch.optim import Optimizer
import torch.nn.functional as F
import torch.nn as nn

Number = Union[float, int]


def breakpoint():
    import pdb

    pdb.set_trace()


class BaseDamper:
    """Damp the noise in the gradient estimate.

    Parameters
    ----------
    model : nn.Module
        The model to train
    dataset : torch.Dataset
        Dataset to use for training
    opt : torch.optim.Optimizer
        The optimizer to use
    loss : callable (function), default=torch.nn.F.nll_loss
        The loss function to use. Must support the reduction keyword. Signature:
        ``loss(output, target, reduction="sum")``.
    initial_batch_size : int, default=1
        Initial batch size
    device : str, default="cpu"
        The device to use.
    max_batch_size : int, float, None, default=None
        The maximum batch size. If the batch size is larger than this
        value, the learning rate is decayed by an appropriate amount.
        If None, will automatically be set to be the size of the
        dataset. Setting to NaN will result in no maximum batch size.
    dwell : int, default=20
        How many model updates should the batch size be held constant?
        This is similar to the "relaxation time" parameter in simulated
        annealing. Setting ``dwell=1`` will mean the batch size will be
        evaluated for every model update.

    Notes
    -----
    By default, this class does not perform any damping (but it's children
    do). If a function needs an instance of BaseDamper, this class can wrap
    any optimizer.

    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        opt: Optimizer,
        loss: Callable = F.nll_loss,
        initial_batch_size: int = 1,
        device: str = "cpu",
        max_batch_size: Optional[int] = None,
        best_train_loss: Optional[float] = None,
        random_state: Optional[int] = None,
        dwell: int=20,
        **kwargs,
    ):
        # Public
        self.initial_batch_size = initial_batch_size
        if max_batch_size is None:
            max_batch_size = len(dataset)
        self.max_batch_size = max_batch_size
        self.dwell = dwell

        # Private
        self._model = model
        self._loss = loss
        self._opt = opt
        self._initial_lr = self._get_lr()
        self._dataset = dataset
        self._device = torch.device(device)
        self._model.to(self._device)
        self.random_state = random_state
        self._rng = np.random.RandomState(seed=random_state)
        self._batch_size = initial_batch_size

        self._meta: Dict[str, Any] = {
            "model_updates": 0,
            "num_examples": 0,
            "batch_loss": None,
            "num_params": sum([m.nelement() for m in model.parameters()]),
            "len_dataset": len(dataset),
            "damper": opt.__class__.__name__.lower(),
            "device_type": self._device.type,
            "loss_name": self._loss.__name__,
            "opt_name": self._opt.__class__.__name__,
        }
        self._meta.update({f"opt_param_{k}": v for k, v in opt.defaults.items()})

    def step(self, **kwargs):
        """Perform an optimization step

        Parameters
        ----------
        kwargs : Dict[str, Any], optional
            Arguments to pass to PyTorch's ``opt.step``
            (e.g., :class:`torch.optim.AdaGrad`)
        """
        start = time()
        mu = self._meta["model_updates"]

        if self._meta["model_updates"] % self.dwell == 0:
            damping = self.damping()
            self._meta["damping_time"] = time() - start
            self._batch_size = int(damping)
        else:
            damping = self._meta["damping"]


        # Is the batch size too large? If so, decay the learning rate
        current_bs = self._batch_size
        max_bs = self.max_batch_size
        if max_bs is not None and current_bs >= max_bs:
            self._set_lr(self._initial_lr * max_bs / current_bs)
            self._batch_size = max_bs

        batch_loss, num_examples = self._step(**kwargs)
        epochs = self._meta["num_examples"] / self._meta["len_dataset"]
        if batch_loss >= 1e6 and epochs > 1:
            raise ConvergenceError(
                f"The model is diverging; batch_loss={batch_loss:0.2e}"
            )

        self._meta["model_updates"] += 1
        self._meta["time"] = time()
        self._meta["step_time"] = time() - start
        self._meta["damping"] = damping
        self._meta["lr_"] = self._get_lr()
        self._meta["num_examples"] += num_examples
        self._meta["batch_loss"] = batch_loss
        self._meta["batch_size"] = self._batch_size

    def damping(self) -> int:
        """Determines how strongly noise in stochastic gradient
        estimate is damped.

        Notes
        -----
        This is the main function for subclasses to overwrite. By
        default, this wraps an optimizer with a static
        ``self.initial_batch_size``. Here's a brief example usage:

            >>> dataset = datasets.MNIST(...)
            >>> model = Net()
            >>> opt = optim.AdaGrad(model.parameters())
            >>> opt = BaseDamper(model, dataset, opt, initial_batch_size=32)
            >>> opt.damping()
            32

        """
        return self.initial_batch_size

    def _get_example_indices(self, batch_size=None):
        if batch_size is None:
            batch_size = self._batch_size
        return self._rng.choice(len(self._dataset), size=batch_size)

    def _get_batch(self, batch_size=None):
        idx = self._get_example_indices()
        data = [self._dataset[i][0] for i in idx]
        data = [d.reshape(-1, *d.size()) for d in data]
        target = [self._dataset[i][1] for i in idx]
        return torch.cat(data), torch.tensor(target)

    def _step(self, **kwargs):
        start = time()

        bs = self._batch_size
        data, target = self._get_batch()
        self._sizes = {"data": data.size(), "target": target.size()}

        data, target = data.to(self._device), target.to(self._device)
        self._opt.zero_grad()

        if bs <= 1024:
            output = self._model(data)
            loss = self._loss(output, target, reduction="sum")
            loss /= len(data)
            loss.backward()
            num_examples = len(data)
            loss_ret = loss.item()
        else:
            num_examples = 0
            Data = torch.split(data, 1024)
            Target = torch.split(target, 1024)
            loss_sum = 0
            for data, target in zip(Data, Target):
                data, target = data.to(self._device), target.to(self._device)
                output = self._model(data)
                loss = self._loss(output, target, reduction="sum")
                loss.backward()

                loss_sum += loss.item()
                num_examples += len(data)
            for _p in self._model.parameters():
                _p.grad /= num_examples
            loss_ret = loss_sum / num_examples

        self._opt.step(**kwargs)
        self._meta["_step_time"] = time() - start
        return loss_ret, num_examples

    def _set_lr(self, lr):
        for group in self._opt.param_groups:
            group["lr"] = lr
        return self._opt

    def _get_lr(self):
        lrs = [group["lr"] for group in self._opt.param_groups]
        assert all(lr == lrs[0] for lr in lrs)
        return lrs[0]

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for this optimzer."""
        return {k: v for k, v in self.__dict__.items() if k[0] != "_"}

    @property
    def meta(self) -> Dict[str, Any]:
        """Get meta information about this optimizer, including number
        of model updates and number of examples processed.
        """
        d = copy(self._meta)
        d.update(self.get_params())
        d["epochs"] = d["num_examples"] / d["len_dataset"]
        return d

    def _get_loss(
        self, dataset: Optional[Dataset] = None, frac: Optional[float] = None,
    ):
        if dataset is None:
            dataset = self._dataset
        num_eg = len(dataset)
        if frac is not None:
            num_eg = int(frac * len(dataset))

        kwargs = (
            {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1000, shuffle=True, **kwargs
        )

        total_loss = 0
        _num_eg = 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self._device), target.to(self._device)
                _num_eg += len(data)
                output = self._model(data)
                loss = self._loss(output, target, reduction="sum")
                total_loss += loss.item()
                if frac is not None and _num_eg >= num_eg:
                    break
        if frac is None:
            assert _num_eg == len(dataset)
        return total_loss / _num_eg


class AdaDamp(BaseDamper):
    def __init__(self, *args, approx_loss=False, **kwargs):
        self.approx_loss = approx_loss
        super().__init__(*args, **kwargs)
        self._meta["damper"] = "adadamp"
        self._meta["_last_batch_losses"] = [None] * 10

    def damping(self) -> int:
        r"""Adaptively damp the noise depending on the current loss with

        .. math::

           B_k = \left\lceil B_0\frac{F(x_0) - F^\star}{F(x_k) - F^\star}\right\rceil

        .. warning::

           This batch size is expensive to compute. It requires evaluating the entire loss function :math:`F`. Use of
           :class:`~PadaDamp` is recommended.

        """
        if not self.approx_loss:
            loss = self._get_loss()
        else:
            _loss = self._meta["batch_loss"]
            _blosses = self._meta["_last_batch_losses"]
            self._meta["_last_batch_losses"] = [_loss, *_blosses[:-1]]
            if any(_ is None for _ in self._meta["_last_batch_losses"]):
                loss = _loss
            else:
                losses = self._meta["_last_batch_losses"]
                loss = np.mean(losses)
            if loss is None or self._meta["batch_size"] <= 25:
                loss = self._get_loss(frac=0.1)
            if loss >= 1e6:
                raise ConvergenceError(f"loss with approx_loss too high ({loss:0.2e})")
            loss *= 0.95

        if self._meta["model_updates"] == 0:
            self._meta["_initial_loss"] = loss
        self._meta["_complete_loss"] = loss
        if np.isnan(loss):
            return 1
        initial_loss = self._meta["_initial_loss"]
        if self._meta.get("best_train_loss", None) is not None:
            initial_loss -= self._meta["best_train_loss"]
            loss -= self._meta["best_train_loss"]
        return _ceil(self.initial_batch_size * initial_loss / loss)


class PadaDamp(BaseDamper):
    r"""
    Parameters
    ----------
    args : list
        Passed to BaseDamper
    batch_growth_rate : float
        The rate to increase the damping by. That is, set the batch size
        to be

        .. math::

           B_k = B_0 \lceil \textrm{rate}\cdot k \rceil

        after the model is updated :math:`k` times.
    kwargs : dict
        Passed to BaseDamper

    Notes
    -----
    The number of epochs is

    .. math::

        uB_0 + \sum_{i=1}^u \lceil \textrm{rate} \cdot k\rceil

    for :math:`u` model updates.

    .. note::

       This class is only appropriate for non-convex and convex loss
       functions. It is not appropriate for strongly convex loss or PL
       functions.

    """

    def __init__(self, *args, batch_growth_rate=None, **kwargs):
        self.batch_growth_rate = batch_growth_rate
        super().__init__(*args, **kwargs)
        self._meta["damper"] = "padadamp"

    def damping(self) -> int:
        r"""Approximate AdaDamp with less computation via

        .. math::

            B_k = B_0 + \lceil \textrm{rate}\cdot k\rceil

        where k is the number of model updates.
        """
        k = self.meta["model_updates"]
        bs = self.initial_batch_size + _ceil(self.batch_growth_rate * k)
        return bs


class GeoDamp(BaseDamper):
    def __init__(self, *args, dampingdelay=5, dampingfactor=2, **kwargs):
        self.dampingdelay = dampingdelay
        self.dampingfactor = dampingfactor
        super().__init__(*args, **kwargs)
        self._meta["damper"] = "geodamp"

    def damping(self) -> int:
        """Set the batch size to increase by ``dampingfactor`` every
        ``dampingdelay`` epochs.
        """
        assert self.dampingfactor >= 1
        epochs = self.meta["num_examples"] / self.meta["len_dataset"]
        factor = self.dampingfactor ** (epochs // self.dampingdelay)
        return self.initial_batch_size * factor


class GeoDampLR(GeoDamp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta["damper"] = "geodamplr"
        self._last_factor = None
        self.max_batch_size = self.initial_batch_size

    def damping(self) -> int:
        """Set the learning rate to decrease by ``dampingfactor`` every
        ``dampingdelay`` epochs.
        """
        return super().damping()


class CntsDampLR(BaseDamper):
    def __init__(self, *args, dampingfactor=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta["damper"] = "cntsdamplr"
        self.dampingfactor = dampingfactor
        self.max_batch_size = self.initial_batch_size

    def damping(self) -> int:
        """Decay the learning rate by :math:`1/k` after :math:`k` model updates.
        """
        k = self._meta["model_updates"]
        bs = np.round(self.initial_batch_size + 1 + self.dampingfactor * (k + 1))
        return bs


class GradientDescent(BaseDamper):
    """This class performs full gradient descent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._meta["damper"] = "gd"

    def _get_example_indices(self, batch_size=None):
        return list(range(len(self._dataset)))

    def damping(self) -> int:
        """ """
        return self._meta["len_dataset"]


def _ceil(x):
    return int(x) + 1


class ConvergenceError(Exception):
    pass
