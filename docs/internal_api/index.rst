.. currentmodule:: uxarray

############
Internal API
############

This page shows already-implemented Uxarray internal API functions. You can also
check the draft `UXarray API
<https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
documentation to see the tentative whole API and let us know if you have any feedback!


UxDataset
=========
The ``uxarray.UxDataset`` class inherits from ``xarray.Dataset``. Below is a list of
features explicitly added to work on Unstructured Grids.

Class
-----
.. autosummary::
   :toctree: _autosummary

   UxDataset


Attributes
----------
.. autosummary::
   :toctree: _autosummary

   UxDataset._source_datasets
   UxDataset._uxgrid

Methods
-------
.. autosummary::
   :toctree: _autosummary

   UxDataset.__getitem__
   UxDataset.__setitem__
   UxDataset._calculate_binary_op
   UxDataset._construct_dataarray
   UxDataset._construct_direct
   UxDataset._copy
   UxDataset._replace



UxDataArray
===========
The ``uxarray.UxDataArray`` class inherits from ``xarray.DataArray``. Below is a list of
features explicitly added to work on Unstructured Grids.

Class
-----
.. autosummary::
   :toctree: _autosummary

   UxDataArray

Attributes
----------
.. autosummary::
   :toctree: _autosummary

   UxDataArray._uxgrid

Methods
-------
.. autosummary::
   :toctree: _autosummary

   UxDataArray._construct_direct
   UxDataArray._copy
   UxDataArray._replace
   UxDataArray._face_centered
   UxDataArray._node_centered

Grid
===========

Class
----------
.. autosummary::
   :toctree: _autosummary

   Grid

Operators
---------
.. autosummary::
   :toctree: _autosummary

   Grid.__eq__
   Grid.__ne__


Helpers
=======

Connectivity
------------
.. autosummary::
   :toctree: _autosummary

   grid.connectivity._face_nodes_to_sparse_matrix
   grid.connectivity._replace_fill_values
   grid.connectivity._build_nNodes_per_face
   grid.connectivity._build_edge_node_connectivity
   grid.connectivity._build_face_edges_connectivity
   grid.connectivity._build_node_faces_connectivity

Geometry
--------
.. autosummary::
   :toctree: _autosummary

   grid.geometry._build_polygon_shells
   grid.geometry._build_corrected_polygon_shells
   grid.geometry._build_antimeridian_face_indices
   grid.geometry._grid_to_polygon_geodataframe
   grid.geometry._grid_to_matplotlib_polycollection
   grid.geometry._grid_to_matplotlib_linecollection
   grid.geometry._pole_point_inside_polygon
   grid.geometry._classify_polygon_location
   grid.geometry._check_intersection

Coordinates
-----------
.. autosummary::
   :toctree: _autosummary

   grid.coordinates._get_lonlat_from_xyz
   grid.coordinates._get_xyz_from_lonlat
   grid.coordinates._populate_cartesian_xyz_coord
   grid.coordinates._populate_lonlat_coord
   grid.coordinates._populate_centroid_coord
   grid.coordinates._construct_xyz_centroids


Lines
-----
.. autosummary::
   :toctree: _autosummary

   grid.lines._angle_of_2_vectors

Utils
-----
.. autosummary::
   :toctree: _autosummary

   grid.utils._fmms
   grid.utils._newton_raphson_solver_for_gca_constLat
   grid.utils._inv_jacobian

Grid Parsing and Encoding
=========================

UGRID
-----
.. autosummary::
   :toctree: _autosummary

   io._ugrid._read_ugrid
   io._ugrid._encode_ugrid
   io._ugrid._standardize_fill_value
   io._ugrid._is_ugrid
   io._ugrid._validate_minimum_ugrid

MPAS
----
.. autosummary::
   :toctree: _autosummary

   io._mpas._read_mpas
   io._mpas._primal_to_ugrid
   io._mpas._dual_to_ugrid
   io._mpas._set_global_attrs
   io._mpas._replace_padding
   io._mpas._replace_zeros
   io._mpas.__to_zero_index


Exodus
---------
.. autosummary::
   :toctree: _autosummary

   io._exodus._read_exodus
   io._exodus._encode_exodus
   io._exodus._get_element_type

SCRIP
-----
.. autosummary::
   :toctree: _autosummary

   io._scrip._to_ugrid
   io._scrip._read_scrip
   io._scrip._encode_scrip


Shapefile
---------
.. autosummary::
   :toctree: _autosummary

   io._shapefile._read_shpfile

Vertices
--------
.. autosummary::
   :toctree: _autosummary

   io._vertices._read_face_vertices

Utils
-----
.. autosummary::
   :toctree: _autosummary

   io.utils._parse_grid_type

Core Utils
----------
.. autosummary::
   :toctree: _autosummary

   core.utils._map_dims_to_ugrid
