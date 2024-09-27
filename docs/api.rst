.. currentmodule:: uxarray

.. _api:

#############
API reference
#############

This page provides an auto-generated summary of UXarray's API. For more details
and examples, refer to the relevant chapters in the main part of the
documentation.

TODO:
See also: :ref:`public api`

High-Level Functions
===================

.. autosummary::
   :toctree: generated/

   open_grid
   open_dataset
   open_dataset


Grid
====


Constructor
-----------

.. autosummary::
   :toctree: generated/

   Grid

I/O & Conversion
----------------

.. autosummary::
   :toctree: generated/

   Grid.from_dataset
   Grid.from_file
   Grid.from_topology
   Grid.to_xarray
   Grid.to_geodataframe
   Grid.to_polycollection
   Grid.to_linecollection
   Grid.encode_as

Indexing
--------
.. autosummary::
   :toctree: generated/

   Grid.isel

Dimensions
----------
.. autosummary::
   :toctree: generated/

   Grid.dims
   Grid.sizes
   Grid.n_node
   Grid.n_edge
   Grid.n_face
   Grid.n_max_face_nodes
   Grid.n_max_face_edges
   Grid.n_max_face_faces
   Grid.n_max_edge_edges
   Grid.n_max_node_faces
   Grid.n_max_node_edges
   Grid.n_nodes_per_face

Spherical Coordinates
---------------------

.. autosummary::
   :toctree: generated/

   Grid.node_lon
   Grid.node_lat
   Grid.edge_lon
   Grid.edge_lat
   Grid.face_lon
   Grid.face_lat

Cartesian Coordinates
---------------------

.. autosummary::
   :toctree: generated/

   Grid.node_x
   Grid.node_y
   Grid.node_z
   Grid.edge_x
   Grid.edge_y
   Grid.edge_z
   Grid.face_x
   Grid.face_y
   Grid.face_z

Connectivity
------------
.. autosummary::
   :toctree: generated/

   Grid.connectivity
   Grid.face_node_connectivity
   Grid.face_edge_connectivity
   Grid.face_face_connectivity
   Grid.edge_node_connectivity
   Grid.edge_edge_connectivity
   Grid.edge_face_connectivity
   Grid.node_node_connectivity
   Grid.node_edge_connectivity
   Grid.node_face_connectivity

Descriptors
-----------
.. autosummary::
   :toctree: generated/

   Grid.descriptors
   Grid.face_areas
   Grid.bounds
   Grid.edge_node_distances
   Grid.edge_face_distances
   Grid.antimeridian_face_indices
   Grid.hole_edge_indices
   Grid.face_jacobian

Attributes
----------
.. autosummary::
   :toctree: generated/

   Grid.attrs

Methods
-------
.. autosummary::
   :toctree: generated/

   Grid.copy
   Grid.chunk
   Grid.validate
   Grid.compute_face_areas
   Grid.calculate_total_face_area
   Grid.normalize_cartesian_coordinates





Plotting
========

Grid
----

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Grid.plot

UxDataArray
-----------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   UxDataArray.plot

UxDataset
---------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   UxDataset.plot




Subsetting
==========

Grid
----

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Grid.subset
   Grid.subset.nearest_neighbor
   Grid.subset.bounding_box
   Grid.subset.bounding_circle


UxDataArray
-----------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   UxDataArray.subset
   UxDataArray.subset.nearest_neighbor
   UxDataArray.subset.bounding_box
   UxDataArray.subset.bounding_circle


Remapping
=========

UxDataArray
-----------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   UxDataArray.remap
   UxDataArray.remap.nearest_neighbor
   UxDataArray.remap.inverse_distance_weighted

UxDataset
---------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   UxDataset.remap
   UxDataset.remap.nearest_neighbor
   UxDataset.remap.inverse_distance_weighted


Calculus Operators
==================

.. autosummary::
   :toctree: generated/

   UxDataArray.integrate
   UxDataArray.gradient


Spherical Geometry
==================

Intersections
-------------

.. autosummary::
   :toctree: generated/

   grid.intersections.gca_gca_intersection
   grid.intersections.gca_constLat_intersection

Arcs
----

.. autosummary::
   :toctree: generated/

   grid.arcs.in_between
   grid.arcs.point_within_gca
   grid.arcs.extreme_gca_latitude


Accurate Computing
==================

.. autosummary::
   :toctree: generated/

   utils.computing.cross_fma
   utils.computing.dot_fma
