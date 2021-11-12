| CI           | [![GitHub Workflow Status][github-ci-badge]][github-ci-link] [![GitHub Workflow Status][github-conda-build-badge]][github-conda-build-link] [![Code Coverage Status][codecov-badge]][codecov-link] |
| :----------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **Docs**     |                                                                    [![Documentation Status][rtd-badge]][rtd-link]                                                                    |
| **Package**  |                                                         [![Conda][conda-badge]][conda-link] [![PyPI][pypi-badge]][pypi-link]                                                         |
| **License**  |                                                                        [![License][license-badge]][repo-link]                                                                        |
| **Citing**   |                                                                              [![DOI][doi-badge]][doi-link]                                                                           |



Uxarray aims to address the geoscience community need for tools that enable standard data analysis techniques to operate directly on unstructured grid data.  These utility functions were inspired by discussion on the [Xarray GitHub Repository](https://github.com/pydata/xarray/issues/4222).

Uxarray will provide Xarray styled funtions to better read in and use unstructured grid datasets that follow standard conventions, including UGRID, SCRIP, Exodus and shapefile formats.  This effort is a result of the collaboration between Project Raijin (NCAR and Pennsylvania State University) and the SEATS Project (Argonne National Laboratory, UC Davis, and Lawrence Livermore National Laboratory).

Uxarray is implemented in pure Python and does not explicitly contain or require any compiled code. This makes Uxarray more
accessible to the general Python community. Any contributions to this repository in pure Python are welcome and documentation
for contribution guidelines can be found when clicking `New Issue` under the `Issues` tab in the Uxarray repository.

# Intended Functionality for Grids

* Support for reading and writing UGRID, SCRIP and Exodus formatted grids.
* Support for reading and writing shapefiles.
* Support for arbitrary structured and unstructured grids on the sphere, including latitude-longitude grids, grids with only partial coverage of the sphere, and grids with concave faces.
* Support for finite volume and finite element outputs.
* Support for edges that are either great circle arc or lines of constant latitude.
* Calculation of face areas, centroids, and bounding latitude-longitude boxes.
* Calculation of supermeshes (consisting of grid lines from two input grids).

# Intended Functionality for DataArrays on Grids

* Regridding of data between unstructured grids.
* Global and regional integration of fields, including zonal averages.
* Application of calculus operations, including divergence, curl, Laplacian and gradient.
* Snapshots and composites following particular features.

# Documentation

[Uxarray Documentation](https://uxarray.readthedocs.io/en/latest)

[Uxarray Contributor’s Guide](https://uxarray.readthedocs.io/en/latest/contributing.html)

[Uxarray Installation](https://uxarray.readthedocs.io/en/latest/installation.html)

[Project Raijin Homepage](https://raijin.ucar.edu/)

[Project Raijin Contributor's Guide](https://raijin.ucar.edu/contributing.html)

[SEATS Project Homepage]()

[SEATs Project Contributor's Guide]()

# Citing Uxarray

If you'd like to cite our work, please follow [How to cite
Uxarray](https://uxarray.readthedocs.io/en/latest/citation.html).



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
[doi-badge]: https://zenodo.org/badge/421447986.svg
[doi-link]: https://zenodo.org/badge/latestdoi/421447986
[repo-link]: https://github.com/UXARRAY/uxarray

# Support

|   |   |
| :-----------: | :----------: |
| <img src="logos/NSF_logo.png" alt="NSF Logo" width="200"/> | Project Raijin, entitled "Collaborative Research: EarthCube Capabilities: Raijin: Community Geoscience Analysis Tools for Unstructured Mesh Data", was awarded by NSF 21-515 EarthCube (Award Number (FAIN): 2126458) on 08/19/2021. The award period of performance has a start date of 09/01/2021 and end date of 08/31/2024. |
| <img src="logos/DOE_Office_of_Science_logo.png" alt="DOE Logo" width="200"/> | SEATS is funded by the Regional and Global Modeling and Analysis (RGMA) program area in the U.S. Department of Energy (DOE) Earth and Environmental System Modeling Program which is part of the Earth and Environmental Systems Sciences Division of the Office of Biological and Environmental Research in DOE’s Office of Science. |

# Partners

|   |   |
| :-----------: | :----------: |
| <img src="logos/EarthCube_logo.png" alt="EarthCube Logo" width="200"/> | EarthCube aims to transform the conduct of geosciences research by developing and maintaining a well-connected and facile environment that improves access, sharing, visualization, and analysis of data and related resources. |
| <img src="logos/PANGEO_logo.png" alt="PANGEO Logo" width="200"/> | Pangeo supports collaborative efforts to develop software and infrastructure to enable Big Data geoscience research. |
