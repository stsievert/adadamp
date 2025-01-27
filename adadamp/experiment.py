from typing import List, Dict, Tuple, Any, Union, Optional, Callable
import itertools
from time import time

import numpy as np
import pandas as pd
import torch.nn as nn
import torch

from .damping import AdaDamp, GeoDamp, PadaDamp, BaseDamper, ConvergenceError

Number = Union[int, float]


def breakpoint():
    import pdb

    pdb.set_trace()

def _clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, float):
        if 10 <= x:
            if np.allclose(x, int(x)):
                return str(int(x))
            return f"{x:0.1f}"
        if 1 <= x < 10:
            return f"{x:0.2f}"
        if 0.01 < x <= 1:
            return f"{x:0.3f}"
        if x < 0.01:
            return f"{x:0.3e}"
    return x

def run(
    model=None,
    opt=None,
    train_set=None,
    test_set=None,
    args=None,
    test_freq: Optional[Number] = None,
    train_stats: bool = True,
    verbose: bool = False,
    device: str = "cpu",
):
    kwargs = {"num_workers": 0, "pin_memory": True} if "cuda" in device.type else {}
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, **kwargs)
    train_test_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1000, **kwargs
    )

    data = []
    train_data = []
    for k in itertools.count():
        test_kwargs = dict(model=model, loss=opt._loss, device=device)
        _train_stats = {}
        if train_stats:
            _train_stats = test(loader=train_test_loader, prefix="train", **test_kwargs)
        test_stats = test(loader=test_loader, prefix="test", **test_kwargs)
        data.append(
            {"epoch_time": time(), **args, **opt.meta, **_train_stats, **test_stats}
        )
        if False:#verbose:
            _s = {
                k: v
                for k, v in data[-1].items()
                if k in [
                    #"damper",
                    #"lr_",
                    "model_updates",
                    "epochs",
                    "damping",
                    "batch_size",
                    #"best_train_loss",
                    #"test_accuracy",
                    #"train_accuracy",
                    #"test_loss",
                    "train_loss",
                ]
            }
            print([v for _, v in _s.items()])
            print(_clean(_s))
        epoch = data[-1]["epochs"]
        mu = data[-1]["model_updates"]
        if epoch >= args["epochs"]:
            break
        try:
            model, opt, epoch_meta, epoch_data = train(
                model,
                opt,
                verbose=args["verbose"],
                epochs=1 if test_freq is None or epoch > 5 else test_freq,
            )
        except ConvergenceError as e:
            print(e)
            break
        train_data += epoch_data
        data[-1].update(epoch_meta)
    return data, train_data


def train(
    model: nn.Module,
    opt: BaseDamper,
    verbose: Optional[Union[int, bool]] = None,
    epochs=1,
) -> Tuple[nn.Module, BaseDamper, Dict[str, Any], List[Dict]]:
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
    if verbose:
        verbose = int(verbose) if isinstance(verbose, bool) else verbose
        print_eg = int(len(opt._dataset) / verbose)
    start_examples = opt._meta["num_examples"]
    old_examples = opt._meta["num_examples"]
    data = []
    _loop_start = time()
    while True:
        num_examples_so_far = opt._meta["num_examples"] - start_examples
        if num_examples_so_far >= epochs * len(opt._dataset):
            break
        opt.step()
        data.append(opt.meta)
        if False:#verbose:# and opt._meta["num_examples"] >= old_examples + print_eg:
            _epochs = opt._meta["num_examples"] / opt._meta["len_dataset"]
            show = ["model_updates", "damping", "batch_size_"]
            print(f"{_epochs:0.2f}", _clean([opt._meta[k] for k in show]))
            old_examples = opt._meta["num_examples"]
    meta = {
        "_epochs": epochs,
        "_num_examples": num_examples_so_far,
        "_train_time": time() - _loop_start,
    }
    return model, opt, meta, data


def test(
    model=None, loss=None, loader=None, device: str = "cpu", prefix=""
):
    assert isinstance(device.type, str)

    def _test(model):
        test_loss = 0
        correct = 0
        _device = device
        model = model.to(_device)
        for data, target in loader:
            data, target = data.to(_device), target.to(_device)
            output = model(data)
            test_loss += loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            if "mse" in loss.__name__:
                continue
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(loader.dataset)
        acc = correct / len(loader.dataset)
        return {"loss": test_loss, "accuracy": acc}

    ret = {"loss": 0}
    model.eval()
    with torch.no_grad():
        ret = _test(model)
        ret.update({"device": device, "prefix": prefix})
    return {f"{prefix}_{k}": v for k, v in ret.items()}
