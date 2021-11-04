| CI           | [![GitHub Workflow Status][github-ci-badge]][github-ci-link] [![GitHub Workflow Status][github-conda-build-badge]][github-conda-build-link] [![Code Coverage Status][codecov-badge]][codecov-link] |
| :----------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **Docs**     |                                                                    [![Documentation Status][rtd-badge]][rtd-link]                                                                    |
| **Package**  |                                                         [![Conda][conda-badge]][conda-link] [![PyPI][pypi-badge]][pypi-link]                                                         |
| **License**  |                                                                        [![License][license-badge]][repo-link]                                                                        |
<!-- | **Citing**   |                                                                              [![DOI][doi-badge]][doi-link]                                                                           | -->



Uxarray aims to address the geoscience community need for utilities that can handle 2D and 3D unstructured grid datasets.
These utility functions were inspired by discussion on the [Xarray GitHub Repository](https://github.com/pydata/xarray/issues/4222).
Uxarray will provide Xarray styled funtions to better read in and use unstructured grid datasets that follow UGRID conventions.
This effort is a result of the collaboration between Project Raijin (NCAR and Pennsylvania State University)
and SEATS Project (Argonne National Laboratory, UC Davis, and Lawrence Livermore National Laboratory).

Uxarray is implemented in pure Python and does not explicitly contain or require any compiled code. This makes Uxarray more
accessible to the general Python community. Any contributions to this repository in pure Python are welcome and documentation
for contribution guidelines can be found when clicking `New Issue` under the `Issues` tab in the Uxarray repository.

# Documentation

[Uxarray Documentation](https://uxarray.readthedocs.io/en/latest)

[Uxarray Contributor’s Guide](https://uxarray.readthedocs.io/en/latest/contributing.html)

[Uxarray Installation](https://uxarray.readthedocs.io/en/latest/installation.html)

[Project Raijin Homepage](https://raijin.ucar.edu/)

[Project Raijin Contributor's Guide](https://raijin.ucar.edu/contributing.html)

[SEATS Project Homepage]()

[SEATs Project Contributor's Guide]()

# Citing Uxarray

Cite Uxarray using the following text:

**UXARRAY Organization. (Year).
Uxarray (Uxarray version \<version\>) [Software].
Project Raijin & Project SEATS. https://uxarray.readthedocs.io/en/latest/.**

Update the Uxarray version and year as appropriate. For example:

**UXARRAY Organization. (2021).
Uxarray (version 0.0.1) [Software].
Project Raijin & Project SEATS. https://uxarray.readthedocs.io/en/latest/.**

For further information, please refer to
[Project Raijin homepage - Citation](https://raijin.ucar.edu/).


[github-ci-badge]: https://img.shields.io/github/workflow/status/UXARRAY/uxarray/CI?label=CI&logo=github&style=for-the-badge
[github-conda-build-badge]: https://img.shields.io/github/workflow/status/UXARRAY/uxarray/build_test?label=conda-builds&logo=github&style=for-the-badge
[github-ci-link]: https://github.com/UXARRAY/uxarray/actions?query=workflow%3ACI
[github-conda-build-link]: https://github.com/UXARRAY/uxarray/actions?query=workflow%3Abuild_test
[codecov-badge]: https://img.shields.io/codecov/c/github/UXARRAY/uxarray.svg?logo=codecov&style=for-the-badge
[codecov-link]: https://codecov.io/gh/UXARRAY/uxarray
[rtd-badge]: https://img.shields.io/readthedocs/uxarray/latest.svg?style=for-the-badge
[rtd-link]: https://uxarray.readthedocs.io/en/latest/?badge=latest
[pypi-badge]: https://img.shields.io/pypi/v/uxarray?logo=pypi&style=for-the-badge
[pypi-link]: https://pypi.org/project/uxarray
[conda-badge]: https://img.shields.io/conda/vn/UXARRAY/uxarray?logo=anaconda&style=for-the-badge
[conda-link]: https://anaconda.org/conda-forge/uxarray
[license-badge]: https://img.shields.io/github/license/UXARRAY/uxarray?style=for-the-badge
<!---[doi-badge]: https://img.shields.io/badge/DOI-10.5065%2Fa8pp--4358-brightgreen?style=for-the-badge --->
<!---[doi-link]: https://doi.org/10.5065/a8pp-4358 --->
[repo-link]: https://github.com/UXARRAY/uxarray
