


[comment]: <> (<img src='https://raw.githubusercontent.com/UXARRAY/uxarray/a2d893cf597dd1a6e775f9dad029c662c64a39c7/docs/_static/images/logos/uxarray_logo_quad_tri.svg' width='400'>)




| CI           | [![GitHub Workflow Status][github-ci-badge]][github-ci-link] [![Code Coverage Status][codecov-badge]][codecov-link]          |
| :----------- | :--------------------------------------------------------------------------------------------------------------------------: |
| **Docs**     |                                                                    [![Documentation Status][rtd-badge]][rtd-link]            |
| **Package**  |                                                         [![Conda][conda-badge]][conda-link] [![PyPI][pypi-badge]][pypi-link] |
| **License**  |                                                                        [![License][license-badge]][repo-link]                |
| **Citing**   |                                                                              [![DOI][doi-badge]][doi-link]                   |



UXarray aims to address the geoscience community's need for tools that enable
standard data analysis techniques to operate directly on unstructured grid
data. UXarray provides Xarray-styled functionality to better read in and use
unstructured grid datasets that follow standard conventions, including UGRID,
MPAS, SCRIP, ESMF, and Exodus grid formats. This effort is a result of the
collaboration between Project Raijin (NSF NCAR and Pennsylvania State University)
and the SEATS Project (Argonne National Laboratory, UC Davis, and Lawrence
Livermore National Laboratory). The UXarray team welcomes community
members to become part of this collaboration at any level of contribution.

UXarray is implemented in pure Python and does not explicitly contain or require
any compiled code. This makes UXarray more accessible to the general Python
community. Any contributions to this repository in pure Python are welcome and
documentation for contribution guidelines can be found when clicking `New Issue`
under the `Issues` tab in the UXarray repository.

## Why is the name "UXarray"?

We have created UXarray based on [Xarray](https://docs.xarray.dev/en/stable/)
(via inheritance of Xarray Dataset and DataArray classes), a Pangeo ecosystem package
commonly-used for structured grids recognition, to support reading and
recognizing unstructured grid model outputs. We picked the name "UXarray"
(pronounced "you-ex-array") and preferred to capitalize the first two letters to
emphasize it builds upon Xarray for Unstructured grids.

## UXarray Functionality

The following intended functionality has been inspired by discussions with
members of the scientific community, within the SEATS Project and Project
Raijin, and on several community platforms such as [Xarray GitHub
Repository](https://github.com/pydata/xarray/issues/4222). The UXarray team
is receptive to additional functionality requests.

## Intended Functionality for Grids

* Support for reading and writing UGRID, SCRIP ESMF, and Exodus formatted grids.
* Support for reading and writing shapefiles.
* Support for arbitrary structured and unstructured grids on the sphere,
  including latitude-longitude grids, grids with only partial coverage of
  the sphere, and grids with concave faces.
* Support for finite volume and finite element outputs.
* Support for edges that are either great circle arcs or lines of constant
  latitude.
* Calculation of face areas, centroids, and bounding latitude-longitude boxes.
* Triangular decompositions.
* Calculation of supermeshes (consisting of grid lines from two input grids).

## Intended Functionality for DataArrays on Grids

* Regridding of data between unstructured grids.
* Global and regional integration of fields, including zonal averages.
* Application of calculus operations, including divergence, curl, Laplacian
  and gradient.
* Snapshots and composites following particular features.

## Documentation

[UXarray Documentation](https://uxarray.readthedocs.io/en/latest)

[Contributor’s Guide](https://uxarray.readthedocs.io/en/latest/contributing.html)

[Installation](https://uxarray.readthedocs.io/en/latest/installation.html)

[Project Raijin Homepage](https://raijin.ucar.edu/)

[SEATS Project Homepage](https://seatstandards.org)

## Citing UXarray

If you'd like to cite our work, please follow [How to cite
UXarray](https://uxarray.readthedocs.io/en/latest/citation.html).

## Support

<table>
  <tr>
    <td><a href="https://www.nsf.gov/"><img src="docs/_static/images/logos/NSF_logo.png" alt="NSF Logo" width="400"/></a></td>
    <td>Project Raijin, entitled "Collaborative Research: EarthCube Capabilities: Raijin: Community Geoscience Analysis Tools for Unstructured Mesh Data", was awarded by NSF 21-515 EarthCube (Award Number (FAIN): 2126458) on 08/19/2021. The award period of performance has a start date of 09/01/2021 and end date of 08/31/2024.</td>
  </tr>
</table>

<table>
  <tr>
    <td><a href="https://www.energy.gov/science/office-science"><img src="docs/_static/images/logos/DOE_vertical.png" alt="DOE Logo" width="1200"/></a></td>
    <td>SEATS is funded by the Regional and Global Modeling and Analysis (RGMA) program area in the U.S. Department of Energy (DOE) Earth and Environmental System Modeling Program which is part of the Earth and Environmental Systems Sciences Division of the Office of Biological and Environmental Research in DOE’s Office of Science.</td>
  </tr>
</table>

<table>
  <tr>
    <td><a href="https://www.earthcube.org/"><img src="docs/_static/images/logos/EarthCube_logo.png" alt="EarthCube Logo" width="300"/></a></td>
    <td><a href="https://www.earthcube.org/">EarthCube</a> aims to transform the conduct of geosciences research by developing and maintaining a well-connected and facile environment that improves access, sharing, visualization, and analysis of data and related resources.</td>
  </tr>
</table>

<table>
  <tr>
    <td><a href="https://pangeo.io/"><img src="docs/_static/images/logos/PANGEO_logo.png" alt="PANGEO Logo" width="400"/></a></td>
    <td><a href="https://pangeo.io/">Pangeo</a> supports collaborative efforts to develop software and infrastructure to enable Big Data geoscience research.</td>
  </tr>
</table>



[github-ci-badge]: https://img.shields.io/github/actions/workflow/status/UXARRAY/uxarray/ci.yml?branch=main&label=CI&logo=github&style=for-the-badge
[github-ci-link]: https://github.com/UXARRAY/uxarray/actions?query=workflow%3ACI
[codecov-badge]: https://img.shields.io/codecov/c/github/UXARRAY/uxarray.svg?logo=codecov&style=for-the-badge
[codecov-link]: https://codecov.io/gh/UXARRAY/uxarray
[rtd-badge]: https://img.shields.io/readthedocs/uxarray/latest.svg?style=for-the-badge
[rtd-link]: https://uxarray.readthedocs.io/en/latest/?badge=latest
[pypi-badge]: https://img.shields.io/pypi/v/uxarray?logo=pypi&style=for-the-badge
[pypi-link]: https://pypi.org/project/uxarray
[conda-badge]: https://img.shields.io/conda/vn/conda-forge/uxarray
[conda-link]: https://anaconda.org/conda-forge/uxarray
[license-badge]: https://img.shields.io/github/license/UXARRAY/uxarray?style=for-the-badge
[doi-badge]: https://zenodo.org/badge/421447986.svg
[doi-link]: https://zenodo.org/badge/latestdoi/421447986
[repo-link]: https://github.com/UXARRAY/uxarray
