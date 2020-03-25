from typing import Callable, Dict, Any, Union, Optional, List

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

IntArray = Union[List[int], np.ndarray, torch.Tensor]


def gradient(
    inputs,
    targets,
    *,
    model: nn.Module,
    loss: Callable,
    device: torch.device = torch.device("cpu"),
    idx: Optional[IntArray] = None,
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
    if idx is not None:
        inputs = inputs[idx]
        targets = targets[idx]
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)
    loss = loss(outputs, targets, reduction="sum")
    loss.backward()
    grads = {k: v.grad for k, v in model.named_parameters()}
    return {"_num_data": len(outputs), "_loss": loss.item(), **grads}
