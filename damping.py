from typing import Callable, Union, Dict, Any, Tuple
from pprint import pprint
from copy import copy

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
        **kwargs,
    ):
        """
        Damp the noise in the gradient estimate.

        Arguments
        ---------
        dataset : torch.Dataset
            Dataset to use for training
        initial_batch_size : int, default=1
            Initial batch size
        kwargs : dict
            Arguments to pass to the underlying torch.DataLoader

        Notes
        -----
        By default, this class does not perform any damping (but it's children
        do). If a function needs an instance of BaseDamper, this class can wrap
        any optimizer.

        """
        self.params = {
            "initial_batch_size": initial_batch_size,
            "opt_params": opt.defaults,
            "device": device,
            "loss": loss.__name__,
            "max_batch_size": max_batch_size,
        }
        self._meta: Dict[str, Any] = {
            "model_updates": 0,
            "num_examples": 0,
            "batch_loss": None,
            "num_params": sum([m.nelement() for m in model.parameters()]),
            "len_dataset": len(dataset),
        }

        self.opt = opt
        self.dataset = dataset
        self.loss = loss
        sampler = RandomSampler(dataset, replacement=True)
        self.loader = DataLoader(dataset, sampler=sampler, drop_last=True, **kwargs,)
        self.model = model
        self.param_groups = self.opt.param_groups
        self.device = torch.device(device)

    def step(self, **kwargs):
        damping = self.damping()
        self.loader.batch_sampler.batch_size = int(
            self.params["initial_batch_size"] * int(damping)
        )

        # Is the batch size too large? If so, decay the learning rate
        current_bs = self.loader.batch_sampler.batch_size
        max_bs = self.params["max_batch_size"]
        if max_bs is not None and current_bs >= max_bs:
            self._set_lr(factor=max_bs / current_bs)
            self.sampler.batch_size = max_bs

        loss, num_examples = self._step(**kwargs)

        self._meta["model_updates"] += 1
        self._meta["damping"] = damping
        self._meta["lr_"] = self._get_lr()
        self._meta["num_examples"] += num_examples
        self._meta["batch_loss"] = loss
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
        return loss.item(), len(data)

    def _set_lr(self, factor=1):
        for group in self.opt.param_groups:
            group["lr"] *= factor
        return self.opt

    def _get_lr(self):
        lrs = [group["lr"] for group in self.opt.param_groups]
        assert all(lr == lrs[0] for lr in lrs)
        return lrs[0]

    @property
    def meta(self):
        d = copy(self._meta)
        d.update(self.params)
        return d


class AdaDamp(BaseDamper):
    def damping(self):
        return 1


class PadaDamp(BaseDamper):
    def damping(self):
        return 1


class GeoDamp(BaseDamper):
    def damping(self):
        return 1


def train(
    model: nn.Module, opt: BaseDamper, print_freq: Union[int, float, None] = 4, epochs=1
):
    """
    Function to train for at least one epoch.

    Arguments
    ---------
    model : nn.Module
        PyTorch model.
    opt : Union[AdaDamp, PadaDamp]
        Optimizer. Must be a subclass of BaseDamper
    print_freq : int, float, None, default=4
        Frequency to print. Prints approximately every ``1 / print_freq``
        fraction of the dataset; setting ``print_freq == 10`` will mean it
        prints 10 times per epoch.

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
    if print_freq is not None:
        print_eg = int(len(opt.dataset) / print_freq)
    start_examples = opt._meta["num_examples"]
    old_examples = opt._meta["num_examples"]
    while True:
        if opt._meta["num_examples"] - start_examples >= len(opt.dataset):
            d = opt._meta["num_examples"] - start_examples
            break
        opt.step()
        if (
            print_freq is not None
            and opt._meta["num_examples"] >= old_examples + print_eg
        ):
            pprint(opt._meta)
            old_examples = opt._meta["num_examples"]
    return model, opt
