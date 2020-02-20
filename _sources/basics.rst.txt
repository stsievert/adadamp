Getting Started
===============

Installation
------------

From PyPI
^^^^^^^^^

Prerequisites: a working installation of PyTorch >= 1.1. Install details can
be found on their home page: https://pytorch.org.


.. code:: shell

   pip install adadamp

From Source
^^^^^^^^^^^

There are no prerequisites for this method.

.. code:: shell

   $ git clone https://github.com/stsievert/adadamp
   $ cd adadamp
   $ conda env create -f adadamp.yaml
   $ conda activate adadamp
   $ python setup.py install

Basic Usage
-----------

Let's make small modifications to `PyTorch's MNIST example`_. This example
usage will be trimmed for brevity. The complete script can be found on GitHub
at `test_mnist.py`_.

.. _test_mnist.py: https://github.com/stsievert/adadamp/blob/master/test_mnist.py

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

Now, let's create the loss function to optimize, alongside the optimizer:

.. code:: python

   import torch.optim as optim
   import torch.nn.functional as F

   _optimizer = optim.SGD(model.parameters(), lr=args.lr)
   loss = F.nll_loss

Again, the optimizer is a pretty standard definition. Use of SGD is recommended
and backed by mathematics. Use of momentum and Nesterov acceleration is likely
beneficial even though there's no backing mathematics.

Let's use :class:`~adadamp.PadaDamp`, which will grow the batch size and is an
approximation to the firmly grounded but impractical :class:`~adadamp.AdaDamp`:

.. code:: python

   from adadamp import PadaDamp

   optimizer = PadaDamp(
       model=model,
       dataset=train_set,
       opt=_optimizer,
       loss=loss,
       device="cpu",
       batch_growth_rate=0.01,
       initial_batch_size=32,
       max_batch_size=1024,
   )

This ``optimizer`` is a drop in replacement for any of PyTorch's optimizers
like :class:`torch.optim.SGD` or :class:`torch.optim.Adagrad`. This means that
we can use it in our (custom) training functions by calling
``optimizer.step()``.

However, it might be easier to use the built-in train/test functions:

.. code:: python

   from adadamp.experiment import train, test

   for epoch in range(1, args.epochs + 1):
       train(model=model, opt=optimizer)
       data = test(model=model, loss=loss, dataset=test_set)
       print(data)

These ``train`` and ``test`` functions are small modifications from the
functions in `PyTorch's MNIST example`_.

.. _PyTorch's MNIST example: https://github.com/pytorch/examples/blob/e9e76722dad4f4569651a8d67ca1d10607db58f9/mnist/main.py

