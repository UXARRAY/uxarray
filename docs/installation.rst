Installation
============

This installation guide includes only the Uxarray installation instructions. Please
refer to `Uxarray Contributor's Guide <https://uxarray.readthedocs.io/en/latest/contributing.html>`_
for detailed information about how to contribute to the uxarray project.

Installing Uxarray via Conda
----------------------------

The easiest way to install Uxarray along with its dependencies is via
`Conda <http://conda.pydata.org/docs/>`_::

    conda install -c conda-forge uxarray

Note that the Conda package manager automatically installs all `required`
dependencies of Uxarray, meaning it is not necessary to explicitly install
Xarray or other required packages when installing Uxarray.

If you are interested in learning more about how Conda environments work, please
visit the `managing environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
page of the Conda documentation.

Installing Uxarray via PyPI
---------------------------

An alternative to Conda is using pip::

    pip install uxarray

Installing Uxarray from source (Github)
---------------------------------------

Installing Uxarray from source code is a fairly straightforward task, but
doing so should not be necessary for most users. If you `are` interested in
installing Uxarray from source, you will first need to get the latest version
of the code::

    git clone https://github.com/UXARRAY/uxarray.git
    cd uxarray

Required dependencies for installing and testing Uxarray
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following packages should be installed (in your active conda
environment)::

    - Python 3.7+
    - `pytest <https://docs.pytest.org/en/stable/>`_  (For tests only)
    - `xarray <http://xarray.pydata.org/en/stable/>`_

If you don't have these packages installed, the next section describes
how to setup a conda environment with them.

Creating a Conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Uxarray source code includes a conda environment definition file
(:code:`environment.yml`) in the :code:`/ci` folder under the root
directory that can be used to create a development environment
containing all of the packages required to build Uxarray. The
following commands should work on Windows, Linux, and macOS to create
and activate a new conda environment from that file::

    conda env create -f ci/environment.yml
    conda activate uxarray_build

Installing from source
^^^^^^^^^^^^^^^^^^^^^^

Once the dependencies listed above are installed, you can install
Uxarray with running the following command from the root-directory::

    pip install .

For compatibility purposes, we strongly recommend using Conda to
configure your build environment as described above.

Testing Uxarray source
^^^^^^^^^^^^^^^^^^^^^^

A Uxarray code base can be tested from the root directory of the source
code repository using the following command (Explicit installation of the
`pytest <https://docs.pytest.org/en/stable/>`_ package may be required, please
see above)::

    pytest test
