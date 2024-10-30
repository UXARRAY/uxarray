<img src="https://github.com/UXARRAY/uxarray/raw/main/docs/_static/images/logos/uxarray_logo_h_dark.svg" data-canonical-src="https://github.com/UXARRAY/uxarray/raw/main/docs/_static/images/logos/uxarray_logo_h_dark.svg" width="450"/><br>


-----------------

# Xarray extension for unstructured climate and global weather data
|    |    |
| --- | --- |
| **Build Status**             | [![GitHub Workflow Status][github-ci-badge]][github-ci-link] [![CI Upstream](https://github.com/UXARRAY/uxarray/actions/workflows/upstream-dev-ci.yml/badge.svg)](https://github.com/UXARRAY/uxarray/actions/workflows/upstream-dev-ci.yml) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/UXARRAY/uxarray/main.svg)](https://results.pre-commit.ci/latest/github/UXARRAY/uxarray/main) |
| **Code Coverage**             |  [![Code Coverage Status][codecov-badge]][codecov-link] |
| **Docs**       |                                   [![Documentation Status][rtd-badge]][rtd-link]                                    |
| **Benchmarks** |                                       [![ASV Repostory][asv-badge]][asv-link]                                       |
| **Releases**    |                        ![Github release](https://img.shields.io/github/release/UXARRAY/uxarray.svg?label=tag&colorB=11ccbb) [![Conda][conda-badge]][conda-link] [![PyPI][pypi-badge]][pypi-link]                         |
| **License**    |                                       [![License][license-badge]][repo-link]                                        |
| **Citing**     |                                            [![DOI][doi-badge]][doi-link]                                            |

-----------------

## What is it?

UXarray aims to address the geoscience community's need for tools that enable
standard data analysis techniques to operate directly on unstructured grid
data. UXarray provides Xarray-styled functionality to better read in and use
unstructured grid datasets that follow standard conventions, including UGRID,
MPAS, ICON, SCRIP, ESMF, FESOM2, and Exodus grid formats. This effort is a result of the
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
(pronounced "you-ex-array"), with the "U" representing unstructured grids.

## Features

* ``Grid`` class for storing grid information and providing grid-specific functionality
  * Support for reading UGRID, MPAS, ESMF, ICON, GEOS-CS, SCRIP, and EXODUS grid formats
  * Support for reader structured (i.e. latitude longitude) grids
* Extension of xarray's ``DataArray`` and ``Dataset`` classe to support unstructured grid operations
  * ``uxarray.UxDataArray`` inherits ``xarray.DataArray`` and is attached to a ``Grid`` instance through the ``.uxgrid`` accessor
  * ``uxarray.UxDataset`` inherits ``xarray.Dataset`` and is attached to a ``Grid`` instance through the ``.uxgrid`` accessor
* Extension of xarray's ``open_dataset`` and ``open_mfdataset`` methods to support reading grid and data files
* Plotting
  * Native visualization of unstructured grids written using the ``hvPlot`` package
  * Support for grid topology visualization (i.e. exploring the geometry of a grid) and data visualization (i.e. data mapped to each face)
* Subsetting
  * Ability to select arbitrary regions of a grid using different selection methods (nearest neighbor, bounding circle, bounding box)
* Remapping
  * Support for nearest neighbor and inverse distance weighted unstructured to unstructured remapping
* Topological Aggregations
  * Perform aggregations within different topology elements using connectivity information
* Mathematical Operators
  * Support for Integral, Difference, and Gradient calculations


## Intended Features

The following intended features have been inspired by discussions with
members of the scientific community, within the SEATS Project and Project
Raijin, and on several community platforms such as [Xarray GitHub
Repository](https://github.com/pydata/xarray/issues/4222). The UXarray team
is receptive to additional functionality requests.

* Support for finite volume and finite element outputs.
* Triangular decompositions.
* Calculation of supermeshes (consisting of grid lines from two input grids).
* Snapshots and composites following particular features.

## Documentation

[UXarray Documentation](https://uxarray.readthedocs.io/en/latest)

[Contributor’s Guide](https://uxarray.readthedocs.io/en/latest/contributing.html)

[Installation](https://uxarray.readthedocs.io/en/latest/getting-started/installation.html)

[Project Raijin Homepage](https://raijin.ucar.edu/)

[SEATS Project Homepage](https://seatstandards.org)

## Contributors

Thank you to all of our contributors!

[![Contributors](https://contrib.rocks/image?repo=UXARRAY/uxarray)](https://github.com/UXARRAY/uxarray/graphs/contributors)

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

[github-ci-badge]: https://github.com/UXARRAY/uxarray/actions/workflows/ci.yml/badge.svg
[github-ci-link]: https://github.com/UXARRAY/uxarray/actions/workflows/ci.yml
[codecov-badge]: https://codecov.io/github/UXARRAY/uxarray/graph/badge.svg?token=lPxJeGNxE0
[codecov-link]: https://codecov.io/github/UXARRAY/uxarray
[rtd-badge]: https://readthedocs.org/projects/uxarray/badge/?version=latest
[rtd-link]: https://uxarray.readthedocs.io/en/latest/?badge=latest
[pypi-badge]: https://img.shields.io/pypi/v/uxarray.svg
[pypi-link]: https://pypi.python.org/pypi/uxarray/
[conda-badge]: https://img.shields.io/conda/vn/conda-forge/uxarray
[conda-link]: https://anaconda.org/conda-forge/uxarray
[license-badge]: https://img.shields.io/github/license/UXARRAY/uxarray
[asv-badge]: https://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat
[asv-link]: https://uxarray.github.io/uxarray-asv/
[doi-badge]: https://zenodo.org/badge/421447986.svg
[doi-link]: https://zenodo.org/badge/latestdoi/421447986
[repo-link]: https://github.com/UXARRAY/uxarray
