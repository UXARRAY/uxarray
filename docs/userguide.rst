.. currentmodule:: uxarray

.. _userguide:

User Guide
==========

The User Guide is the primary resource documenting key concepts and functionality within UXarray.
For newcomers, a gentle introduction to UXarray can be found in the `Getting Started <quickstart.rst>`_
guide and examples of UXarray in action can be found in our `Gallery <gallery.rst>`_

Introductory Guides
-------------------

These user guides provide the necessary background information for understanding concepts in UXarray.

`Terminology <user-guide/terminology.rst>`_
 Core concepts and terminology for working with unstructured grids

`Conventions <user-guide/representation.rst>`_
 Overview of the UGRID conventions and how they are used within UXarray

`Supported Models & Grid Formats <user-guide/grid-formats.rst>`_
 Overview of supported models and grid formats

Core Guides
------------

These user guides provide detailed explanations of the core functionality in UXarray.

`Data Structures <user-guide/data-structures.ipynb>`_
 Core data structures for working with unstructured grid and data files

`Custom Grid Topology <user-guide/custom-grid.ipynb>`_
 Create a Grid from custom Grid topology and convert existing Xarray data structures to UXarray.

`Loading Data using Dask <user-guide/parallel-load-ux-with-dask.ipynb>`_
 Read data with chunking and/or in parallel

`Plotting <user-guide/plotting.ipynb>`_
 Visualize unstructured grid datasets using UXarray's plotting API

`Plotting with Matplotlib <user-guide/mpl.ipynb>`_
 Use Matplotlib for plotting with PolyCollection and LineCollection

`Advanced Plotting Techniques <user-guide/advanced-plotting.ipynb>`_
 Deeper dive into getting the most out of UXarray's plotting functionality

`Subsetting <user-guide/subset.ipynb>`_
 Select specific regions of a grid

`Spatial Hashing <user-guide/spatial-hashing.ipynb>`_
 Use spatial hashing to locate the faces a list of points reside in.

`Cross-Sections <user-guide/cross-sections.ipynb>`_
 Select cross-sections of a grid

`Zonal Means <user-guide/zonal-average.ipynb>`_
 Compute the zonal averages across lines of constant latitude

`Remapping <user-guide/remapping.ipynb>`_
 Remap (a.k.a Regrid) between unstructured grids

`Topological Aggregations <user-guide/topological-aggregations.ipynb>`_
 Aggregate data across grid dimensions

`Weighted Mean <user-guide/weighted_mean.ipynb>`_
 Compute the weighted average

`Calculus Operators <user-guide/calculus.ipynb>`_
 Apply calculus operators (gradient, integral) on unstructured grid data

`Tree Structures <user-guide/tree_structures.ipynb>`_
 Data structures for nearest neighbor queries

`Face Area Calculations <user-guide/area_calc.ipynb>`_
 Methods for computing the area of each face

`Structured Grids <user-guide/structured.ipynb>`_
 Loading structured (latitude-longitude) grids

`Representing Point Data <user-guide/from-points.ipynb>`_
 Create grids from unstructured point data

`Dual Mesh Construction <user-guide/dual-mesh.ipynb>`_
 Construct the Dual Mesh of an unstructured grid


Supplementary Guides
--------------------

These user guides provide additional details about specific features in UXarray.

`Working with HEALPix Grids <user-guide/healpix.ipynb>`_
 Use UXarray with HEALPix

`Compatibility with HoloViz Tools <user-guide/holoviz.ipynb>`_
 Use UXarray with HoloViz tools

`Reading & Working with Geometry Files <user-guide/from_file.ipynb>`_
 Load and work with geometry files (i.e. Shapefile, GeoJSON)

.. toctree::
   :hidden:

   user-guide/terminology.rst
   user-guide/representation.rst
   user-guide/grid-formats.rst
   user-guide/data-structures.ipynb
   user-guide/parallel-load-ux-with-dask.ipynb
   user-guide/plotting.ipynb
   user-guide/mpl.ipynb
   user-guide/advanced-plotting.ipynb
   user-guide/subset.ipynb
   user-guide/cross-sections.ipynb
   user-guide/zonal-average.ipynb
   user-guide/remapping.ipynb
   user-guide/topological-aggregations.ipynb
   user-guide/weighted_mean.ipynb
   user-guide/calculus.ipynb
   user-guide/tree_structures.ipynb
   user-guide/area_calc.ipynb
   user-guide/dual-mesh.ipynb
   user-guide/structured.ipynb
   user-guide/from-points.ipynb
   user-guide/healpix.ipynb
   user-guide/holoviz.ipynb
   user-guide/from_file.ipynb
