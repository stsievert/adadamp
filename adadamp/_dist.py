from typing import Callable, Dict, Any, Union, Optional, List

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

IntArray = Union[List[int], np.ndarray, torch.Tensor]


def gradient(
    train_set,
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
    # data_target = None
    # if idx is not None:
    #     data_target = [train_set for i in idx] 
    # else:
    #     data_target = [train_set for i in len(data_target)]

    # inputs = [d[0].reshape(-1, *d[0].size()) for d in data_target] 
    # target = [d[1] for d in data_target]

    inputs = [pt[0] for pt in train_set] # client.scatter(train_data)
    targets  = [pt[1] for pt in train_set]  #client.scatter(train_lbl)

    # inputs = [pt[0] for pt in train_set] # client.scatter(train_data)
    # targets  = [pt[1] for pt in train_set]  #client.scatter(train_lbl)

    if idx is not None:
        # TypeError: only integer scalar arrays can be converted to a scalar index
        inputs = inputs[idx]
        targets = targets[idx]
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)
    loss = loss(outputs, targets, reduction="sum")
    loss.backward()
    grads = {k: v.grad for k, v in model.named_parameters()}
    return {"_num_data": len(outputs), "_loss": loss.item(), **grads}
