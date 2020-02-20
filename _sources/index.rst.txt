.. adadamp documentation master file, created by
   sphinx-quickstart on Tue Feb 18 18:48:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to adadamp's documentation!
===================================

The optimization underlying machine learning has to handle big data.  It does
this by approximating the loss function's gradient with a few examples, aka the
batch size. This is nearly universal among different optimization methods.

How should the batch size be selected? Small batch sizes will result in highly
variable gradient estimates but can compute many model updates for a given
computation budget. Large batch sizes will result in more precise gradient
estimate, but can't compute as many model updates for the same computation budget.

This package provides a method founded in math to balance these two extremes.
Balancing these two extremes means that **training will finished more
quickly**. More detail can be found in the `Mathematical underpinning` and
`Experiments` sections.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   basics
   math
   experiments
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
