.. currentmodule:: uxarray

======================
Overview: Why UXarray?
======================
UXarray aims to address the geoscience community need for tools that enable standard
data analysis techniques to operate directly on unstructured grids. It extends upon
and inherits from the commonly used Xarray Python package to provide a powerful and
familiar interface for working with unstructured grids in Python. UXarray provides
Xarray styled functions to better read in and use unstructured grid datasets that
follow standard conventions, including UGRID, MPAS, SCRIP, and Exodus formats.


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

Specifically, our core functionality is build around two-dimension
Unstructured Grids as defined by the 2D Flexible Mesh Topology in the
UGRID conventions, which can contain a mix of triangles, quadrilaterals, or
other geometric faces.





Core Data Structures
====================

The functionality of UXarray is built around three core data structures which provide
an Unstructured Grid aware implementation of many Xarray functions and use cases.

* ``Grid`` is used to represent our Unstructured Grid, housing grid-specific methods
  and topology variables.
* ``UxDataset`` inherits from the ``xarray.Dataset`` class, providing much of the same
  functionality but extended to operate on Unstructured Grids. Other than new and
  overloaded methods, it is linked to a ``Grid`` object through the use of a class
  property (``UxDataset.uxgrid``) to provide a grid-aware implementation. An instance
  of ``UxDataset`` can be thought of as a collection of Data Variables that reside on
  some Unstructured Grid as defined in the ``uxgrid`` property.
* ``UxDataArray`` similarly inherits from the ``xarray.DataArray`` class and contains
  a ``Grid`` property (``UxDataArray.uxgrid``) just like ``UxDataset``.

Core Functionality
====================
In addition to providing a way to load in and interface with Unstructured Grids, we
also aim to provide computational and analysis operators that directly operate on
Unstructured Grids.

The list of currently implemented operators can be found in the
`User API <https://uxarray.readthedocs.io/en/latest/user_api/index.html>`_
documentation.

Get involved in the `Prioritization of Uxarray analysis
operators <https://github.com/UXARRAY/uxarray/discussions/46>`_ to be released in
the future!
