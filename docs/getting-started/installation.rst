.. currentmodule:: uxarray
.. _installation:

============
Installation
============

UXarray is built **on top of** `Xarray <https://docs.xarray.dev/en/latest/getting-started-guide/installing.html#installation>`__,
so we **strongly** recommend becoming familiar with Xarray’s installation
process and dependencies first.

For details on contributing, see the :doc:`UXarray Contributor’s Guide <contributing>`.


Installing with Conda (recommended)
-----------------------------------

UXarray itself is a pure Python package, but its dependencies are not.
The easiest way to get everything installed is to use conda.
To install UXarray with its recommended dependencies using the conda command line tool:


.. code-block:: bash

   conda install -c conda-forge uxarray

.. note::

   Conda automatically installs Xarray and every other required
   dependency (including non‑Python libraries).

Installing with pip
-------------------
.. code-block:: bash

   pip install uxarray

This installs the *minimal* required dependencies. UXarray also provides optional extras:

.. code-block:: bash

   pip install "uxarray[dev]"       # development tools
   pip install "uxarray[complete]"  # all optional features

A complete list of extras lives in the ``[project.optional-dependencies]``
section of our `pyproject.toml <https://github.com/UXARRAY/uxarray/blob/main/pyproject.toml>`_


Installing from source
----------------------
Installing from source is intended mainly for developers.

#. **Clone the repo**

   .. code-block:: bash

      git clone https://github.com/UXARRAY/uxarray.git
      cd uxarray

#. **Create a dev environment**

   A ready-made file is provided at ``ci/environment.yml``:

   .. code-block:: bash

      conda env create -f ci/environment.yml
      conda activate uxarray_build

#. **Install UXarray**

   .. code-block:: bash

      pip install .

#. **Run the test suite**

   .. code-block:: bash

      pytest test

Verifying your installation
---------------------------

After installing UXarray, you can verify the installation by running the following in a Python shell or script:

.. code-block:: python

    import uxarray as ux

    print(ux.__version__)

This should print the installed version of UXarray without errors.
