Basic Usage
===========

Let's walk through `PyTorch's MNIST example`_. This example
usage will be trimmed for brevity. The complete script can be found on GitHub
at `test_mnist_dask.py`_.

.. _test_mnist_dask.py: https://github.com/stsievert/adadamp/blob/master/test_mnist_dask.py

First, let's start by creating the model.

.. code:: python

   import torch
   import torch.nn as nn
   from torchvision import datasets

   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.conv1 = nn.Conv2d(1, 32, 3, 1)
           # etc; unchanged from example

        def forward(self, x):
           # etc; unchanged from example

   model = Net()

This is a standard PyTorch model definition. Now, let's create the dataset:

.. code:: python

   train_set = datasets.MNIST(...)  # internals untouched
   test_set = datasets.MNIST(...)  # internals untouched

Notice a :class:`~torch.utils.data.DataLoader` object is not created (as with
PyTorch), only the dataset that's passed to
:class:`~torch.utils.data.DataLoader`.

Now, let's create our Scikit-learn wrapper with our model, loss and optimizer:


.. code-block:: python

   from adadamp import DaskClassifier
   est = DaskClassifier(
       module=Net,
       loss=nn.NLLLoss,
       optimizer=optim.SGD,
       optimizer__lr=1e-3,
       optimizer__momentum=0.9,
       optimizer__nesterov=True,
       batch_size=256,
       max_epochs=10,
   )

Again, the optimizer is a pretty standard definition. Use of SGD is recommended
and backed by mathematics. Use of momentum and Nesterov acceleration is likely
beneficial even though there's no backing mathematics.

Now, we can feed our data to the ``fit`` method:

.. code-block:: python

   est.fit(train_set)
   acc = est.score(test_set)

If desired, we can also feed our function NumPy arrays or Torch Tensors:

.. code-block:: python

   from sklearn.datasets import make_classification
   X, y = make_classification()
   est.fit(X, y)
   X2, y2 = torch.from_numpy(X), torch.from_numpy(y)
   est.fit(X2, y2)

.. _PyTorch's MNIST example: https://github.com/pytorch/examples/blob/e9e76722dad4f4569651a8d67ca1d10607db58f9/mnist/main.py
