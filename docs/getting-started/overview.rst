.. currentmodule:: uxarray

======================
Overview: Why UXarray?
======================
UXarray aims to address the geoscience community need for tools that enable standard data analysis techniques
to operate directly on unstructured grids. It extends upon and inherits from the commonly used Xarray Python package
to provide a powerful and familiar interface for working with unstructured grids in Python.


Unstructured Grids
==================
The "U" in UXarray stands for "Unstructured Grids". These types of grids differ from typical Structured Grids in
terms of complexity, requiring additional overhead to store and represent the geometry and topology of the grid. However,
these types of grids are extremely flexible and scalable.

UXarray uses the UGRID conventions to represent Unstructured Grids. These conventions are intended to describe how
these grids should be stored within a NetCDF file, with a particular focus on environmental and geoscience
applications. We chose to use a single convention for our grid representation instead of having separate ones for
each grid format, meaning that we encode all supported unstructured grid formats in the UGRID conventions at
the data loading step.

Specifically, we represented our two-dimensional Unstructured Grids using a 2D Flexible Mesh topology, which can
contain a mix of triangles, quadrilaterals, or other geometric faces.



Core Data Structures
====================

The functionality of UXarray is build around three core data structures which provide an Unstructured Grid aware
implementation of many Xarray functions and use cases.

* ``Grid`` is used to represent our Unstructured Grid, housing grid-specific methods and topology variables.
* ``UxDataset`` inherits from the ``xarray.Dataset`` class, providing much of the same functionality but extended
  to operate on Unstructured Grids. Other than new and overloaded methods, it is linked to a ``Grid`` object through
  the use of an accessor (``UxDataset.uxgrid``) to provide a grid-aware implementation. An instance of ``UxDataset``
  can be thought of as a collection of Data Variables that reside on some Unstructured Grid
  as defined in the ``Grid`` accessor.
* ``UxDataArray`` similarly inherits from the ``xarray.DataArray`` class and contains a the same ``Grid``
  accessor (``UxDataArray.uxgrid``) that is linked to the ``UxDataset``.
  Each data variable stored in a ``UxDataset`` is represented as a ``UxDataArray``

Core Functionality
====================
In addition to providing a way to load in and interface with Unstructured Grids, we also aim to provide
computational and analysis operators that directly operate on Unstructured Grids. Below is a list of currently
implemented and planned operators.

**Implemented**

* Face Area Calculations
* Integration

**Planned**

* Global and Zonal Means (Conservative & Non-Conservative)
* Regridding
* Visualizations
* Gradients, Divergence, Curl, Laplacian
