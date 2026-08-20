.. currentmodule:: uxarray
.. _installation:

============
Installation
============

UXarray is built **on top of** `Xarray <https://docs.xarray.dev/en/latest/getting-started-guide/installing.html#installation>`__,
so we **strongly** recommend becoming familiar with Xarray’s installation
process and dependencies first.

Installing with Conda (recommended)
-----------------------------------

UXarray itself is a pure Python package, but its dependencies are not.
The easiest way to get everything installed is to use conda.
To install UXarray with its recommended dependencies using the conda command line tool:


.. code-block:: bash

   conda install -c conda-forge uxarray

.. note::

   Conda automatically installs Xarray, all other required depdencies,
   and all optional dependencies too (including non‑Python libraries).

Installing with pip
-------------------
.. code-block:: bash

   pip install uxarray

This installs the *minimal* required dependencies. UXarray also provides optional extras:

.. code-block:: bash

   pip install "uxarray[dev]"       # development tools
   pip install "uxarray[geo]"       # geospatial tools (e.g. geopandas, healpix)
   pip install "uxarray[viz]"       # plotting tools (e.g. matplotlib, hvplot)
   pip install "uxarray[complete]"  # all optional features

A complete list of optional dependencies lives in the ``[project.optional-dependencies]``
section of our `pyproject.toml <https://github.com/UXARRAY/uxarray/blob/main/pyproject.toml>`_

Optional extras can also be combined, e.g. ``pip install "uxarray[geo,viz]"`` installs with
all required dependencies plus all optional dependencies for geospatial and plotting tools.


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

      pip install ".[complete]"   # test suite relies on optional dependencies

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
