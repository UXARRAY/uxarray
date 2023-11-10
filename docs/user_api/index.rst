.. currentmodule:: uxarray

########
User API
########

This page shows already-implemented Uxarray user API functions. You can also
check the `UXarray Milestones <https://github.com/UXARRAY/uxarray/milestones>`_ and
`UXarray RoadMap <https://github.com/orgs/UXARRAY/projects/2/views/17>`_ for a high
level understanding of UXarray's future function development milestones and roadmap.
Please let us know if you have any feedback!

UxDataset
=========
A ``xarray.Dataset``-like, multi-dimensional, in memory, array database.
Inherits from ``xarray.Dataset`` and has its own unstructured grid-aware
dataset operators and attributes through the ``uxgrid`` accessor.

Below is a list of features explicitly added to `UxDataset` to work on
Unstructured Grids:

Class
-----
.. autosummary::
   :toctree: _autosummary

   UxDataset

IO
--
.. autosummary::
   :toctree: _autosummary

   open_dataset
   open_mfdataset

Attributes
----------
.. autosummary::
   :toctree: _autosummary

   UxDataset.uxgrid
   UxDataset.source_datasets

Methods
-------
.. autosummary::
   :toctree: _autosummary

   UxDataset.info
   UxDataset.integrate


Remapping
---------
.. autosummary::
   :toctree: _autosummary

   UxDataset.nearest_neighbor_remap

Plotting
--------
.. autosummary::
   :toctree: _autosummary

   UxDataset.plot



UxDataArray
===========
N-dimensional ``xarray.DataArray``-like array. Inherits from `xarray.DataArray`
and has its own unstructured grid-aware array operators and attributes through
the ``uxgrid`` accessor.

Below is a list of features explicitly added to `UxDataset` to work on
Unstructured Grids:

Class
-----
.. autosummary::
   :toctree: _autosummary

   UxDataArray

IO
--
.. autosummary::
   :toctree: _autosummary

   UxDataArray.to_dataset
   UxDataArray.to_geodataframe
   UxDataArray.to_polycollection


Attributes
----------
.. autosummary::
   :toctree: _autosummary
   UxDataArray.uxgrid

Methods
-------
.. autosummary::
   :toctree: _autosummary

   UxDataArray.integrate


Remapping
---------
.. autosummary::
   :toctree: _autosummary

   UxDataArray.nearest_neighbor_remap

Plotting
--------
.. autosummary::
   :toctree: _autosummary

   UxDataArray.plot
   UxDataArray.plot.datashade
   UxDataArray.plot.rasterize




Grid
====
Unstructured grid topology definition to store stores grid topology dimensions,
coordinates, variables and provides grid-specific functions.

Can be used standalone to explore an unstructured grid topology, or can be
seen as the property of ``uxarray.UxDataset`` and ``uxarray.DataArray`` to make
them unstructured grid-aware data sets and arrays.

Class
-----
.. autosummary::
   :toctree: _autosummary

   Grid

IO
--
.. autosummary::
   :toctree: _autosummary

   open_grid
   Grid.from_dataset
   Grid.from_face_vertices
   Grid.to_geodataframe
   Grid.to_polycollection
   Grid.to_linecollection
   Grid.to_shapely_polygons


Methods
-------
.. autosummary::
   :toctree: _autosummary

   Grid.calculate_total_face_area
   Grid.compute_face_areas
   Grid.encode_as
   Grid.get_ball_tree
   Grid.get_kd_tree
   Grid.copy


Dimensions
----------
.. autosummary::
   :toctree: _autosummary

   Grid.n_node
   Grid.n_edge
   Grid.n_face
   Grid.n_max_face_nodes
   Grid.n_max_face_edges
   Grid.n_nodes_per_face

Spherical Coordinates
---------------------
.. autosummary::
   :toctree: _autosummary

   Grid.node_lon
   Grid.node_lat
   Grid.edge_lon
   Grid.edge_lat
   Grid.face_lon
   Grid.face_lat

Cartesian Coordinates
---------------------
.. autosummary::
   :toctree: _autosummary

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
   :toctree: _autosummary

   Grid.face_node_connectivity
   Grid.edge_node_connectivity
   Grid.node_node_connectivity
   Grid.face_edge_connectivity
   Grid.edge_edge_connectivity
   Grid.node_edge_connectivity
   Grid.face_face_connectivity
   Grid.edge_face_connectivity
   Grid.node_face_connectivity

Grid Descriptors
----------------
.. autosummary::
   :toctree: _autosummary

   Grid.face_areas
   Grid.antimeridian_face_indices


Attributes
----------
.. autosummary::
   :toctree: _autosummary

   Grid.grid_spec
   Grid.parsed_attrs


Plotting
--------
.. autosummary::
   :toctree: _autosummary

   Grid.plot



Visualization
=============

Accessors
---------
.. autosummary::
   :toctree: _autosummary

   plot.accessor.GridPlotAccessor
   plot.accessor.UxDataArrayPlotAccessor
   plot.accessor.UxDatasetPlotAccessor

UxDataArray Plotting Methods
----------------------------
.. autosummary::
   :toctree: _autosummary

   plot.dataarray_plot.datashade
   plot.dataarray_plot.rasterize



Nearest Neighbor Data Structures
================================

KDTree
------
.. autosummary::
   :toctree: _autosummary

   grid.neighbors.KDTree
   grid.neighbors.KDTree.query
   grid.neighbors.KDTree.query_radius

BallTree
--------
.. autosummary::
   :toctree: _autosummary

   grid.neighbors.BallTree
   grid.neighbors.BallTree.query
   grid.neighbors.BallTree.query_radius


Helpers
=======

Face Area
----------
.. autosummary::
   :toctree: _autosummary

   grid.area.calculate_face_area
   grid.area.get_all_face_area_from_coords
   grid.area.calculate_spherical_triangle_jacobian
   grid.area.calculate_spherical_triangle_jacobian_barycentric
   grid.area.get_gauss_quadratureDG
   grid.area.get_tri_quadratureDG

Connectivity
------------
.. autosummary::
   :toctree: _autosummary

   grid.connectivity.close_face_nodes

Coordinates
-----------
.. autosummary::
   :toctree: _autosummary

   grid.coordinates.node_lonlat_rad_to_xyz
   grid.coordinates.node_xyz_to_lonlat_rad
   grid.coordinates.normalize_in_place


Arcs
----
.. autosummary::
   :toctree: _autosummary

   grid.arcs.in_between
   grid.arcs.point_within_gca
   grid.arcs.extreme_gca_latitude

Intersections
-------------
.. autosummary::
   :toctree: _autosummary

   grid.intersections.gca_gca_intersection
   grid.intersections.gca_constLat_intersection

Accurate Computing Utils
-----
.. autosummary::
   :toctree: _autosummary

   utils.computing.cross_fma
   utils.computing.dot_fma

Numba
-----
.. autosummary::
   :toctree: _autosummary

   utils.enable_jit_cache
   utils.disable_jit_cache
   utils.enable_jit
   utils.disable_jit
