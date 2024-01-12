.. currentmodule:: uxarray

############
Internal API
############

This page shows already-implemented Uxarray internal API functions. You can also
check the `UXarray Milestones <https://github.com/UXARRAY/uxarray/milestones>`_ and
`UXarray RoadMap <https://github.com/orgs/UXARRAY/projects/2/views/17>`_ for a high
level understanding of UXarray's future function development milestones and roadmap.
Please let us know if you have any feedback!


UxDataset
=========
The ``uxarray.UxDataset`` class inherits from ``xarray.Dataset``. Below is a list of
features explicitly added to work on Unstructured Grids.

Class
-----
.. autosummary::
   :toctree: generated/

   UxDataset


Attributes
----------
.. autosummary::
   :toctree: generated/

   UxDataset._source_datasets
   UxDataset._uxgrid

Methods
-------
.. autosummary::
   :toctree: generated/

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
   :toctree: generated/

   UxDataArray

Attributes
----------
.. autosummary::
   :toctree: generated/

   UxDataArray._uxgrid

Methods
-------
.. autosummary::
   :toctree: generated/

   UxDataArray._construct_direct
   UxDataArray._copy
   UxDataArray._replace
   UxDataArray._face_centered
   UxDataArray._node_centered
   UxDataArray._slice_from_grid

Grid
====

Class
-----
.. autosummary::
   :toctree: generated/

   Grid

Operators
---------
.. autosummary::
   :toctree: generated/

   Grid.__eq__
   Grid.__ne__


Helpers
=======

Connectivity
------------
.. autosummary::
   :toctree: generated/

   grid.connectivity._face_nodes_to_sparse_matrix
   grid.connectivity._replace_fill_values
   grid.connectivity._build_nNodes_per_face
   grid.connectivity._build_edge_node_connectivity
   grid.connectivity._build_face_edges_connectivity
   grid.connectivity._build_node_faces_connectivity
   grid.connectivity._build_edge_face_connectivity
   grid.connectivity._populate_edge_node_connectivity
   grid.connectivity._populate_face_edges_connectivity
   grid.connectivity._populate_node_faces_connectivity
   grid.connectivity._populate_edge_face_connectivity
   grid.connectivity._populate_n_nodes_per_face

Geometry
--------
.. autosummary::
   :toctree: generated/
   grid.geometry._pad_closed_face_nodes
   grid.geometry._build_polygon_shells
   grid.geometry._grid_to_polygon_geodataframe
   grid.geometry._build_geodataframe_without_antimeridian
   grid.geometry._build_geodataframe_with_antimeridian
   grid.geometry._build_corrected_shapely_polygons
   grid.geometry._build_antimeridian_face_indices
   grid.geometry._populate_antimeridian_face_indices
   grid.geometry._build_corrected_polygon_shells
   grid.geometry._grid_to_matplotlib_polycollection
   grid.geometry._grid_to_matplotlib_linecollection
   grid.geometry._pole_point_inside_polygon
   grid.geometry._classify_polygon_location
   grid.geometry._check_intersection

Coordinates
-----------
.. autosummary::
   :toctree: generated/

   grid.coordinates._get_lonlat_from_xyz
   grid.coordinates._get_xyz_from_lonlat
   grid.coordinates._populate_cartesian_xyz_coord
   grid.coordinates._populate_lonlat_coord
   grid.coordinates._populate_centroid_coord
   grid.coordinates._construct_xyz_centroids
   grid.coordinates._set_desired_longitude_range


Arcs
----
.. autosummary::
   :toctree: generated/

   grid.arcs._angle_of_2_vectors
   grid.arcs._angle_of_2_vectors


Utils
-----
.. autosummary::
   :toctree: generated/

   grid.utils._newton_raphson_solver_for_gca_constLat
   grid.utils._inv_jacobian
   grid.utils._get_cartesiain_face_edge_nodes



Validation
----------
.. autosummary::
   :toctree: generated/

   grid.validation._check_connectivity
   grid.validation._check_duplicate_nodes
   grid.validation._check_area

Accurate Computing Utils
------------------------
.. autosummary::
   :toctree: generated/

   utils.computing._err_fmac
   utils.computing._fast_two_mult
   utils.computing._fast_two_sum
   utils.computing._two_sum
   utils.computing._two_prod_fma
   utils.computing._comp_prod_fma
   utils.computing._sum_of_squares_re
   utils.computing._vec_sum
   utils.computing._norm_faithful
   utils.computing._norm_l
   utils.computing._norm_g
   utils.computing._two_square
   utils.computing._acc_sqrt
   utils.computing._split

Remapping
=========

.. autosummary::
   :toctree: generated/

   remap.nearest_neighbor._nearest_neighbor
   remap.nearest_neighbor._nearest_neighbor_uxda
   remap.nearest_neighbor._nearest_neighbor_uxds


Grid Parsing and Encoding
=========================

UGRID
-----
.. autosummary::
   :toctree: generated/

   io._ugrid._read_ugrid
   io._ugrid._encode_ugrid
   io._ugrid._standardize_fill_value
   io._ugrid._is_ugrid
   io._ugrid._validate_minimum_ugrid

MPAS
----
.. autosummary::
   :toctree: generated/

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
   :toctree: generated/

   io._exodus._read_exodus
   io._exodus._encode_exodus
   io._exodus._get_element_type

SCRIP
-----
.. autosummary::
   :toctree: generated/

   io._scrip._to_ugrid
   io._scrip._read_scrip
   io._scrip._encode_scrip


Shapefile
---------
.. autosummary::
   :toctree: generated/

   io._shapefile._read_shpfile

Vertices
--------
.. autosummary::
   :toctree: generated/

   io._vertices._read_face_vertices

Utils
-----
.. autosummary::
   :toctree: generated/

   io.utils._parse_grid_type

Core Utils
----------
.. autosummary::
   :toctree: generated/

   core.utils._map_dims_to_ugrid


Visualization
-------------
.. autosummary::
   :toctree: generated/

   plot.grid_plot._plot_coords_as_points
   plot.dataarray_plot._plot_data_as_points
   plot.dataarray_plot._polygon_raster
   plot.dataarray_plot._point_raster

Slicing
-------
.. autosummary::
   :toctree: generated/

   grid.slice._slice_node_indices
   grid._slice_edge_indices
   grid._slice_face_indices



Subsetting
----------
.. autosummary::
   :toctree: generated/

   subset.grid_accessor.GridSubsetAccessor
   subset.dataarray_accessor.DataArraySubsetAccessor
