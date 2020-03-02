from typing import Callable, Dict, Any, Union

import torch
import torch.nn.functional as F
import torch.nn as nn


def gradient(
    model: nn.Module,
    input,
    target,
    loss: Callable,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Union[torch.Tensor, int]]:
    r"""
    Compute the model gradient for the function ``loss``.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to eval
    data : PyTorch DataSet
    loss : Callabale
        PyTorch loss. Should return the loss with

        .. code:: python

           loss(output, out_, reduction="sum")

    device : PyTorch device (optional)

    Returns
    -------
    grad : Dict[str, Union[Tensor, int]]
        Gradients. This dictionary all keys in model.named_parameters.keys().

    Notes
    -----
    This function computes the gradient of the *sum* of inputs, not the *mean*
    of inputs. Functionally, this means evaluates the gradient of
    :math:`\sum_{i=1}^B l(...)`, not :math:`\frac{1}{n} \sum_{i=1}^B l(...)`
    where `l` is the loss function for a single example.

    """
    output = model(input)
    loss = loss(output, target, reduction="sum")
    loss.backward()
    ret = {k: v.grad for k, v in model.named_parameters()}
    ret["_num_data"] = len(output)
    ret["_loss"] = loss.item()
    return ret
