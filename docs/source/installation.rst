Installation
============

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

Test
----
From the ``adadamp/`` directory, run the following command:

.. code:: shell

   $ pytest

This is easiest if you install from source.
