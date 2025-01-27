from typing import Callable, Dict, Any, Tuple, Set, List, Optional, Union
from copy import copy, deepcopy
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
    random_state : int, optional
        The random state the samples are selected in.

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
        dwell: int = 20,
        **kwargs,
    ):
        # Public
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size or len(dataset)
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
        updates = self._meta["model_updates"]

        damping = self.damping() if updates % int(self.dwell) == 0 else self._meta["damping"]
        self._meta["damping"] = damping

        # Is the batch size too large? If so, decay the learning rate
        batch_size = deepcopy(damping)
        max_bs = self.max_batch_size
        if max_bs is not None and batch_size >= max_bs:
            self._set_lr(self._initial_lr * max_bs / batch_size)
            batch_size = max_bs

        batch_loss, num_examples = self._step(batch_size, **kwargs)
        epochs = self._meta["num_examples"] / self._meta["len_dataset"]
        if (batch_loss >= 3e3 or np.isnan(batch_loss)) and epochs > 1:
            raise ConvergenceError(
                f"The model is diverging; batch_loss={batch_loss:0.2e}"
            )

        self._meta["model_updates"] += 1
        self._meta["time"] = time()
        self._meta["step_time"] = time() - start
        self._meta["lr_"] = self._get_lr()
        self._meta["num_examples"] += num_examples
        self._meta["batch_loss"] = batch_loss
        self._meta["batch_size_"] = batch_size
        extras = self._step_callback(self._model)
        self._meta.update(extras)

    def _step_callback(self, model):
        return {}

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
        if np.isnan(batch_size):
            batch_size = np.inf
        batch_size = min(batch_size, len(self._dataset))
        return self._rng.choice(len(self._dataset), size=int(batch_size), replace=False)

    def _get_batch(self, batch_size=None):
        idx = self._get_example_indices(batch_size=batch_size)
        data_target = [self._dataset[i] for i in idx]
        data = [d[0].reshape(-1, *d[0].size()) for d in data_target]
        target = [d[1] for d in data_target]
        if target[0].ndim == 1:
            t_out = torch.tensor(target)
        elif target[0].ndim == 0:
            t_out = torch.tensor([t.item() for t in target])
        else:
            target2 = [t.reshape(-1, *t[0].size()) for t in target]
            t_out = torch.stack(target2)
        return torch.cat(data), t_out

    def _step(self, batch_size, **kwargs):
        bs = batch_size
        start = time()

        data, target = self._get_batch(batch_size=batch_size)
        self._sizes = {"data": data.size(), "target": target.size()}

        self._opt.zero_grad()

        mbs = 32 * 3
        if bs <= mbs:
            data, target = data.to(self._device), target.to(self._device)
            output = self._model(data)
            loss = self._loss(output, target, reduction="sum")
            loss /= len(data)
            loss.backward()
            num_examples = len(data)
            loss_ret = loss.item()
        else:
            num_examples = 0
            Data = torch.split(data, mbs)
            Target = torch.split(target, mbs)
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
    ) -> float:
        if dataset is None:
            dataset = self._dataset
        num_eg = len(dataset)
        if frac is not None:
            num_eg = int(frac * len(dataset))

        kwargs = (
            {"num_workers": 0, "pin_memory": True} if torch.cuda.is_available() else {}
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
        assert type(total_loss) == float
        return total_loss / _num_eg

    def _get_grads(self, frac=None) -> Tuple[float, List[np.ndarray]]:
        dataset = self._dataset
        kwargs = (
            {"num_workers": 0, "pin_memory": True} if torch.cuda.is_available() else {}
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1000, shuffle=True, **kwargs
        )
        num_eg = len(dataset)
        if frac is not None:
            num_eg = int(frac * len(dataset))

        total_loss = 0
        _num_eg = 0
        for data, target in loader:
            data, target = data.to(self._device), target.to(self._device)
            _num_eg += len(data)
            output = self._model(data)
            loss = self._loss(output, target, reduction="sum")
            loss.backward()
            total_loss += float(loss)

            if frac is not None and _num_eg >= num_eg:
                break
        return total_loss / _num_eg, [p.grad.detach().cpu().numpy() for p in self._model.parameters()]

class AdaDampNN(BaseDamper):
    def __init__(self, *args, noisy=False, approx=False, best_norm2=1, **kwargs):
        self.approx = approx
        self.noisy = noisy
        super().__init__(*args, **kwargs)
        self._meta["damper"] = "adadampnn"
        self._meta["_last_batch_losses"] = [None] * 10
        self._counter = 0

    def damping(self) -> int:
        self._opt.zero_grad()
        if self.approx:
            loss, grads = self._get_grads(frac=0.1)
        else:
            loss, grads = self._get_grads()
        self._opt.zero_grad()

        with torch.no_grad():
            norm2 = sum(np.sum(g**2) for g in grads)
        n_params = sum(g.size for g in grads)
        norm2 /= n_params

        if loss >= 1e4:
            raise ConvergenceError(f"loss too high, ({loss:0.2e})")
        if self._meta["model_updates"] == 0:
            self._meta["_initial"] = norm2
            return self.initial_batch_size

        self._meta["_current"] = norm2

        _initial = self._meta["_initial"]
        _current = self._meta["_current"]
        #print(f"{self._counter} {_initial:0.3f} {_current:0.3f}")#, {bs}")

        #if self._meta.get("best_norm2", None) is not None:
        #    initial_loss -= self._meta["best_norm2"]
        #    loss -= self._meta["best_norm2"]
        bs = _ceil(self.initial_batch_size * _initial / _current)
        #if self.noisy:
        #return max(self.initial_batch_size, bs)
        return bs

class RadaDamp(BaseDamper):
    def __init__(self, *args, fudge=0.01, rho=0.999, fn_class="smooth", **kwargs):
        self.rho = rho
        self.fn_class = fn_class
        self.fudge = fudge
        super().__init__(*args, **kwargs)
        self._meta["damper"] = "radadamp"

    def _step_callback(self, model):
        if self.fn_class != "smooth":
            return {}

        with torch.no_grad():
            norm2 = 0.0
            for p in model.parameters():
                x = torch.norm(p.grad).item()
                assert isinstance(x, (float, np.floating))
                norm2 += float(x) ** 2

        return {"_batch_grad_norm2": norm2, "_batch_grad_norm": np.sqrt(norm2)}

    def step(self, *args, **kwargs):
        limit = 50
        _grad_key = "_batch_grad_norm2"
        if self._meta["model_updates"] == 0:
            grads = self._get_grads()
            norm2 = sum(np.linalg.norm(g) ** 2 for g in grads)
            norm = np.sqrt(norm2)
            self._meta["_initial_norm2"] = copy(norm2)
            self._meta["_batch_grad_norm2"] = copy(norm2)
            self._meta["_batch_grad_norm"] = copy(norm)

            loss = self._get_loss(frac=0.01)
            self._meta["_initial_loss"] = loss

            self._meta["_grad_mavg"] = 0.0
            self._meta["_loss_mavg"] = 0.0

        elif self._meta["model_updates"] < limit:
            # avoid initial large gradients
            self._meta["_grad_mavg"] += copy(self._meta[_grad_key])

            self._meta["_loss_mavg"] += copy(
                self._meta["batch_loss"] or self._meta["_initial_loss"]
            )
        elif self._meta["model_updates"] == limit:
            self._meta["_grad_mavg"] /= self._meta["model_updates"]
            self._meta["_loss_mavg"] /= self._meta["model_updates"]
            init_factor = (self.fudge * self._meta["_loss_mavg"]) + (self._meta["_grad_mavg"])
            self._meta["_initial_factor"] = init_factor
        return super().step(*args, **kwargs)

    @staticmethod
    def _rolling_avg(current, past, rho):
        return (1 - rho) * current + (rho * past)

    def damping(self) -> int:
        limit = 50
        if self._meta["model_updates"] <= limit:
            return self.initial_batch_size
        _grad_key = "_batch_grad_norm2"
        self._meta["_loss_mavg"] = self._rolling_avg(
            self._meta["batch_loss"], self._meta["_loss_mavg"], self.rho
        )
        self._meta["_grad_mavg"] = self._rolling_avg(
            self._meta[_grad_key], self._meta["_grad_mavg"], self.rho
        )

        div = self.fudge * self._meta["_loss_mavg"] + (self._meta["_grad_mavg"])

        _factor = self._meta["_initial_factor"] / div
        factor = max(1, _factor)
        damping = int(self.initial_batch_size * factor)
        return damping


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
            loss = self._get_loss(frac=0.1)
            if loss >= 1e6:
                raise ConvergenceError(f"loss with approx_loss too high ({loss:0.2e})")

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
    """
    Passive AdaDamp

    Parameters
    ----------
    rho : float (default=0.5)
        Memory. High rho means slow adaptation, very stable. Low rho means very
        adaptive, quick reaction.
    """
    def __init__(self, *args, growth_rate=1e-3, **kwargs):
        self.growth_rate = growth_rate
        super().__init__(*args, **kwargs)
        self._meta["damper"] = "padadamp"

    def damping(self) -> int:
        mu = self._meta["model_updates"]
        bs = self.initial_batch_size + self.growth_rate * mu
        return _ceil(bs)

class PrAdaDamp(BaseDamper):
    """
    Practical AdaDamp

    Parameters
    ----------
    rho : float (default=0.5)
        Memory. High rho means slow adaptation, very stable. Low rho means very
        adaptive, quick reaction.
    """
    def __init__(self, *args, reduction: str = "mean", rho: float = 0.5, wait = 10, **kwargs):
        self.reduction = reduction
        self.rho = rho
        self.wait = wait
        super().__init__(*args, **kwargs)
        self._meta["damper"] = "pradadamp"
        self._meta["norm2_hist"] = []

    def _step_callback(self, model):
        with torch.no_grad():
            grads = [x.grad.detach() for x in model.parameters()]
            norms2 = [(g**2).sum() for g in grads]
            norm2 = sum(norms2).item()
        self._meta["norm2_hist"].append(norm2)
        return {}

    def damping(self) -> int:
        if self._meta["model_updates"] == 0:
            return self.initial_batch_size

        norms2 = self._meta["norm2_hist"]
        if self.reduction == "median":
            norm2 = np.median(norms2)
        elif self.reduction == "mean":
            norm2 = np.mean(norms2)
        elif self.reduction == "min":
            norm2 = np.min(norms2)
        elif self.reduction == "max":
            norm2 = np.max(norms2)
        #elif self.reduction == "exponential":
        #    _norm2 = np.mean(norms2)
        #    norm2 = self.rho * (1 - self.norm2) + (1 - self.rho) * _norm2
        else:
            raise ValueError(f"reduction={self.reduction} not recognized")

        if self._meta["model_updates"] <= max(self.dwell, self.wait):
            self._initial = norm2
            self.norm2 = norm2
            return self.initial_batch_size
        if not (0 <= self.rho < 1):
            raise ValueError(f"rho={self.rho} not valid, not in 0 <= rho < 1")

        self.norm2 = self.rho * self.norm2 + (1 - self.rho) * norm2
        self._meta["norm2_hist"] = []

        bs = _ceil(self.initial_batch_size * self._initial / self.norm2)
        return max(bs, self.initial_batch_size)
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


def _ceil(x: float) -> int:
    return int(np.ceil(x).astype(int))


class ConvergenceError(Exception):
    pass
