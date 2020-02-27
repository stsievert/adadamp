.. adadamp documentation master file, created by
   sphinx-quickstart on Tue Feb 18 18:48:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to adadamp's documentation!
===================================

The optimization underlying machine learning has to handle big data. For
example, `the MNIST dataset`_ has 60,000 images in the training set, and that's a
small dataset. By comparison, the popular `ImageNet dataset`_ has over 10 million images.

.. _ImageNet dataset: https://en.wikipedia.org/wiki/ImageNet
.. _the MNIST dataset: https://en.wikipedia.org/wiki/MNIST_database

Machine learning tries to minimize a "loss function" that depends on every
example in the dataset (or conversely, maximize a "quality function").  This
requires the gradient of the loss function, which depends on every data. If
there aren't many examples, "gradient descent" is used or a modification
thereof.

However, most machine learning models require thousands of gradient
computations. To avoid this, machine learning uses `stochastic` gradient
descent (SGD) which approximates the loss function's gradient with a few
examples, aka the batch size. This is nearly universal among different
optimization methods.

How should the batch size be selected? Small batch sizes will result in highly
variable gradient estimates but can compute many model updates for a given
computation budget. Large batch sizes will result in more precise gradient
estimate, but can't compute as many model updates for the same computation budget.

This package provides a method founded in math to balance these two extremes.
Use of this package will result in two benefits, at least with a particular
setup of a distributed system: machine learning **models will be trained more
quickly** than standard SGD (or competing methods). [#flops]_

More detail can be found in the :ref:`sec-math` and :ref:`sec-exps` sections.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   basic-usage
   math
   experiments
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [#flops] Don't worry -- if the distributed setup is not available, this
            package will not require more floating point operations than
            standard SGD.

