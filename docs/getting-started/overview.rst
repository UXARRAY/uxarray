.. currentmodule:: uxarray

============
Why UXarray?
============
UXarray aims to address the geoscience community's need for tools that enable foundational
data analysis techniques to operate directly on unstructured grids. It extends upon and
inherits from the commonly used Xarray Python package to provide a powerful and familiar
interface for working with unstructured grids in Python. UXarray provides Xarray-styled
functions to better read in and use unstructured grid datasets from the most common model
outputs, including MPAS, CAM-SE, ICON, ESMF, GEOS, and HEALPix, which follow the commonly
used formats such as UGRID, SCRIP, and Exodus. Furthermore, UXarray provides basic support
for generating unstructured grid topology from structured grid or point-cloud inputs to
enable model intercomparison workflows.


Unstructured Grids
==================
The "U" in UXarray stands for "Unstructured Grids". These types of grids differ from
typical Structured Grids in terms of complexity, requiring additional overhead to
store and represent their geometry and topology. However, these types of
grids are extremely flexible and scalable.

UXarray uses the `UGRID <http://ugrid-conventions.github.io/ugrid-conventions/>`_
conventions as a
foundation to represent Unstructured Grids. These conventions
are intended to describe how these grids should be stored within a NetCDF file, with
a particular focus on environmental and geoscience applications. We chose to use a
single convention for our grid representation instead of having separate ones for each
grid format, meaning that we encode all supported unstructured grid formats in the
UGRID conventions at the data loading step.

Specifically, our core functionality is built around two-dimensional
Unstructured Grids as defined by the 2D Flexible Mesh Topology in the
UGRID conventions, which can contain a mix of triangles, quadrilaterals, or
other geometric faces.


Core Data Structures
====================

UXarray’s core API revolves around three primary types, which extend Xarray for unstructured-grid workflows:

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - :py:class:`uxarray.Grid`
     - Represents the unstructured grid itself, housing grid-specific methods and topology variables. Encapsulates a :py:class:`xarray.Dataset` for storing the grid definition.

   * - :py:class:`uxarray.UxDataset`
     - Extends :py:class:`xarray.Dataset` to operate on unstructured grids; linked to a :py:class:`~uxarray.Grid` instance via its ``uxgrid`` property.

   * - :py:class:`uxarray.UxDataArray`
     - Similarly extends :py:class:`xarray.DataArray` and exposes a ``uxgrid`` accessor for grid-aware operations.



Core Functionality
==================

In addition to loading and interfacing with Unstructured Grids, UXarray provides
computational and analysis operators that operate directly on those grids. Some of
these include:

* Visualization
* Remapping
* Subsetting
* Cross-Sections
* Aggregations
* Zonal Averaging

A more detailed overview of supported functionality can be found in our
`API Reference <https://uxarray.readthedocs.io/en/latest/api.html>`_
and `User Guide <https://uxarray.readthedocs.io/en/latest/userguide.html>`_
sections.
