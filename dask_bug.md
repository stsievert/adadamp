---
name: Bug Report
about: Create a report to help us improve Dask

---

<!-- 
Thank you for taking the time to file a bug report. 
Please fill in the fields below, deleting the sections that 
don't apply to your issue. You can view the final output
by clicking the preview button above.
Note: This is a comment, and won't appear in the output.
-->

My issue is about errors when Serializing custom PyTorch models with dask. When running the following code snippet, I am able to scatter my custom model to the client, but if I scatter the train set then the model, I get some unexpected behavior and my program crashes.

Note that if you were to replace my custom model with a built in pytorch model such as resnet18, the program will not crash.

#### Reproducing code example:
<!-- 
If you place your code between the triple backticks below, 
it will be rendered as a code block. 
-->

```python
from torchvision.datasets import FashionMNIST
from copy import deepcopy
from distributed import Client
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Example custom model
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 5, stride=1)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    # setup model and net
    client = Client()
    #  model = resnet18()  # code passes if uncommented
    model = Net()
    # get data
    train_set = FashionMNIST("_traindata/fashionmnist/", train=True, download=True)
    # set model to dask, ensure error free
    model_future = client.scatter(deepcopy(model))
    assert True, "Sanity check to make sure reached"
    # Hm... this is where our bug occurs, when scatter the model following our trian set
    ts_f3 = client.scatter(train_set, broadcast=True)
    model_future = client.scatter(deepcopy(model))

```

#### Error message:
<!-- If any, paste the *full* error message inside a code block
as above (starting from line Traceback)
-->

```
distributed.protocol.core - CRITICAL - Failed to deserialize
Traceback (most recent call last):
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/protocol/core.py", line 130, in loads
    value = _deserialize(head, fs, deserializers=deserializers)
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/protocol/serialize.py", line 297, in deserialize
    return loads(header, frames)
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/protocol/serialize.py", line 50, in dask_loads
    loads = dask_deserialize.dispatch(typ)
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/dask/utils.py", line 499, in dispatch
    raise TypeError("No dispatch for {0}".format(cls))
TypeError: No dispatch for <class '__main__.Net'>
distributed.core - ERROR - No dispatch for <class '__main__.Net'>
Traceback (most recent call last):
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/core.py", line 337, in handle_comm
    msg = await comm.read()
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/comm/tcp.py", line 202, in read
    msg = await from_frames(
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/comm/utils.py", line 75, in from_frames
    res = _from_frames()
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/comm/utils.py", line 60, in _from_frames
    return protocol.loads(
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/protocol/core.py", line 130, in loads
    value = _deserialize(head, fs, deserializers=deserializers)
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/protocol/serialize.py", line 297, in deserialize
    return loads(header, frames)
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/protocol/serialize.py", line 50, in dask_loads
    loads = dask_deserialize.dispatch(typ)
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/dask/utils.py", line 499, in dispatch
    raise TypeError("No dispatch for {0}".format(cls))
TypeError: No dispatch for <class '__main__.Net'>
/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/torch/storage.py:34: FutureWarning: pickle support for Storage will be removed in 1.5. Use `torch.save` instead
  warnings.warn("pickle support for Storage will be removed in 1.5. Use `torch.save` instead", FutureWarning)
distributed.core - ERROR - No dispatch for <class '__main__.Net'>
Traceback (most recent call last):
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/core.py", line 403, in handle_comm
    result = await result
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/scheduler.py", line 2797, in scatter
    keys, who_has, nbytes = await scatter_to_workers(
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/utils_comm.py", line 144, in scatter_to_workers
    out = await All(
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/utils.py", line 237, in All
    result = await tasks.next()
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/core.py", line 750, in send_recv_from_rpc
    result = await send_recv(comm=comm, op=key, **kwargs)
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/core.py", line 551, in send_recv
    raise Exception(response["text"])
Exception: No dispatch for <class '__main__.Net'>
Traceback (most recent call last):
  File "dask_torch_mvp.py", line 33, in <module>
    model_future = client.scatter(deepcopy(model))
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/client.py", line 2164, in scatter
    return self.sync(
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/client.py", line 815, in sync
    return sync(
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/utils.py", line 347, in sync
    raise exc.with_traceback(tb)
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/utils.py", line 331, in f
    result[0] = yield future
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/tornado/gen.py", line 735, in run
    value = future.result()
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/client.py", line 2055, in _scatter
    await self.scheduler.scatter(
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/core.py", line 750, in send_recv_from_rpc
    result = await send_recv(comm=comm, op=key, **kwargs)
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/core.py", line 549, in send_recv
    raise exc.with_traceback(tb)
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/core.py", line 403, in handle_comm
    result = await result
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/scheduler.py", line 2797, in scatter
    keys, who_has, nbytes = await scatter_to_workers(
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/utils_comm.py", line 144, in scatter_to_workers
    out = await All(
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/utils.py", line 237, in All
    result = await tasks.next()
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/core.py", line 750, in send_recv_from_rpc
    result = await send_recv(comm=comm, op=key, **kwargs)
  File "/Users/joeholt/miniconda3/envs/adadamp/lib/python3.8/site-packages/distributed/core.py", line 551, in send_recv
    raise Exception(response["text"])
Exception: No dispatch for <class '__main__.Net'>
```

#### Version information:

Dask - 2.16.0

Dask-core - 2.16.0

Distributed - 2.16.0

Numpy - 1.18.1

Python - 3.8.1

PyTorch - 1.5.0
