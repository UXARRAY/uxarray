.. currentmodule:: uxarray

======================
Overview: Why UXarray?
======================

Unstructured Grids
==================
Quick overview of Unstructured Grids and a reference to the UGRID conventions

Core Data Structures
====================

The functionality of UXarray is build around three core data structures which provide a Unstructured Grid aware
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

Goals and Motivation
====================
UXarray aims to address the geoscience community need for tools that enable standard data analysis techniques to operate
directly on unstructured grid data.
