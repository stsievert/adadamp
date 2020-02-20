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
usage will be trimmed for brevity. The complete script can be found at
https://github.com/stsievert/adadamp/blob/master/test_mnist.py.

First, let's start by

.. code:: python

   import torch
   import torch.nn as nn

   class Net(nn.Module):
       def __init__(self):
           super(Net, self).__init__()
           self.conv1 = nn.Conv2d(1, 32, 3, 1)
           # etc; nearly unchanged from example

   model = Net()


Now, let's create our datasets:

.. code:: python

   from torchvision import datasets

   train_set = datasets.MNIST(...)  # internals untouched
   test_set = datasets.MNIST(...)  # internals untouched

Now, let's create the loss function to optimize, alongside the optimizer:

.. code:: python

   import torch.optim as optim
   import torch.nn.functional as F

   _optimizer = optim.SGD(model.parameters(), lr=args.lr)
   loss = F.nll_loss


Let's use :class:`~adadamp.PadaDamp`, which will grow the batch size.

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

This will result in quicker training. Let's do that:

.. code:: python

   from adadamp.experiment import train, test

   for epoch in range(1, args.epochs + 1):
       train(model=model, opt=optimizer)
       data = test(model=model, loss=loss, dataset=test_set)
       print(data)

These ``train`` and ``test`` functions are small modifications. The
:class:`~adadamp.PadaDamp` instance conforms to the `torch.optim.Optimizer`
API, and can be used as a drop-in replacement (besides initialization).

.. _PyTorch's MNIST example: https://github.com/pytorch/examples/blob/e9e76722dad4f4569651a8d67ca1d10607db58f9/mnist/main.py

