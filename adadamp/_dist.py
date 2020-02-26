from typing import Callable, Dict, Any, Union

import torch
import torch.nn.functional as F
import torch.nn as nn

def gradient(
    model: nn.Module,
    input,
    target,
    loss: Callable,
    device = torch.device("cpu"),
) -> Dict[str, Union[torch.Tensor, int]]:
    """
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

    """
    output = model(input)
    loss = loss(output, target, reduction="sum")
    loss.backward()
    ret = {k: v.grad for k, v in model.named_parameters()}
    ret["_num_data"] = len(output)
    return ret

