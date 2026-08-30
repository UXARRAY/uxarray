.. currentmodule:: uxarray
.. _installation:

============
Installation
============

UXarray itself is a pure Python package, but its dependencies are not.
The easiest way to get everything installed is to use conda or pip.

.. admonition:: Conda versus Pip

    Conda installs UXarray with all optional depencies.
    Pip installs only the minimal required dependencies by default, but it
    is also easy to include any/all optional dependencies too, if desired.


Installing with Conda
---------------------

To install UXarray with its recommended dependencies using the conda command line tool:

.. code-block:: bash

   conda install -c conda-forge uxarray


Installing with pip
-------------------
For a lightweight installation with **only required dependencies**:

.. code-block:: bash

   pip install uxarray


For a complete installation which also includes **all optional dependencies**:

.. code-block:: bash

   pip install "uxarray[complete]"


For an installation including **only some optional dependencies**,
consider using one of the following extras:

- ``pip install "uxarray[dev]"`` includes development tools (e.g. pytest, ruff)
- ``pip install "uxarray[geo]"`` includes geospatial packages (e.g. geopandas, healpix)
- ``pip install "uxarray[viz]"`` includes plotting packages (e.g. matplotlib, hvplot)

It is also possible to combine extras, for example:

- ``pip install "uxarray[geo,viz]"`` includes all geospatial and plotting packages.


To see the full lists of which optional dependencies are included with each extra group,
take a look at the ``[project.optional-dependencies]`` section in ``pyproject.toml``:

.. literalinclude:: ../../pyproject.toml
   :language: toml
   :start-at: [project.optional-dependencies]
   :end-before: [project.urls]


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

#. **Optional: run the test suite**

    Running the test suite is a good way to verify that the installation is working correctly.
    It should take roughly a minute to run on modern machines.

   .. code-block:: bash

      pytest test

Verifying your installation
---------------------------

After installing UXarray, you can verify the installation by running the following in a Python shell or script:

.. code-block:: python

    import uxarray as ux

    print(ux.__version__)

This should print the installed version of UXarray without errors.
