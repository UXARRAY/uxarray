Installation
============

This installation guide includes only the Uxarray installation and build instructions.
Please refer to `Uxarray Contributor's Guide <https://github.com/UXARRAY/uxarray>`_ for installation of
the project.

Installing Uxarray via Conda
--------------------------------

The easiest way to install GeoCAT-comp is using
`Conda <http://conda.pydata.org/docs/>`_::

    conda create -n uxarray -c conda-forge uxarray

where "uxarray" is the name of a new conda environment, which can then be
activated using::

    conda activate uxarray

If you somewhat need to make use of other software packages, such as Matplotlib,
Cartopy, Jupyter, etc. with Uxarray, you may wish to install into your :code:`uxarray`
environment.  The following :code:`conda create` command can be used to create a new
:code:`conda` environment that includes some of these additional commonly used Python
packages pre-installed::

    conda create -n uxarray -c conda-forge uxarray matplotlib cartopy jupyter

Alternatively, if you already created a conda environment using the first
command (without the extra packages), you can activate and install the packages
in an existing environment with the following commands::

    conda activate uxarray # or whatever your environment is called
    conda install -c conda-forge matplotlib cartopy jupyter



Also, note that the Conda package manager automatically installs all `required`
dependencies of Uxarray, meaning it is not necessary to explicitly install
Xarraywhen creating an environment and installing Uxarray.

If you are interested in learning more about how Conda environments work, please
visit the `managing environments <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_
page of the Conda documentation.


Building Uxarray from source
--------------------------------

Building Uxarray from source code is a fairly straightforward task, but
doing so should not be necessary for most users. If you `are` interested in
building Uxarray from source, you will need the following packages to be
installed.

Required dependencies for building and testing Uxarray
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    - Python 3.7+
    - `pytest <https://docs.pytest.org/en/stable/>`_  (For tests only)
    - `xarray <http://xarray.pydata.org/en/stable/>`_



How to create a Conda environment for building Uxarray
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Uxarray source code includes a conda environment definition file in
the :code:`/build_envs` folder under the root directory that can be used to create a
development environment containing all of the packages required to build Uxarray.
The file :code:`environment.yml` is intended to be used on Linux systems and macOS.
The following commands should work on both Linux and macOS::

    conda env create -f build_envs/environment.yml
    conda activate uxarray_build


Installing Uxarray
^^^^^^^^^^^^^^^^^^^^^^

Once the dependencies listed above are installed, you can install Uxarray
with running the following command from the root-directory::

    pip install .

For compatibility purposes, we strongly recommend using Conda to
configure your build environment as described above.


Testing a Uxarray build
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A Uxarray build can be tested from the root directory of the source code
repository using the following command (Explicit installation of the
`pytest <https://docs.pytest.org/en/stable/>`_ package may be required, please
see above)::

    pytest test
