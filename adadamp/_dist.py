from typing import Callable, Dict, Any, Union, Optional, List
from torch.autograd import Variable

import torch
from copy import copy
import torch.nn as nn
import numpy as np

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
    # the reason for loading the dataset like this is to avoid
    # a performance hit when manually pulling from a PyTorch data
    # loader
    #
    # See: https://github.com/stsievert/adadamp/pull/2#discussion_r416187089
    data_target = [train_set[i] for i in idx]
    _inputs = [d[0].reshape(-1, *d[0].size()) for d in data_target]
    _targets = [d[1] for d in data_target]
    inputs = torch.cat(_inputs)
    targets = torch.tensor(_targets)

    outputs = model(inputs)

    _loss = loss(outputs, targets, reduction="sum")
    _loss.backward()
    grads = {k: v.grad for k, v in model.named_parameters()}
    return {"_num_data": len(outputs), "_loss": _loss.item(), **grads}
