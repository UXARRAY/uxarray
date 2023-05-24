.. currentmodule:: uxarray

Internal API
============

This page shows already-implemented Uxarray internal API functions. You can also
check the draft `Uxarray API
<https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
documentation to see the tentative whole API and let us know if you have any feedback!

Grid Methods
--------------------
.. currentmodule:: uxarray.core.grid.Grid
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

<<<<<<< HEAD
   __init_grid_var_names__
   __from_ds__
   __from_vert__
   __init_grid_var_attrs__
   _build_edge_node_connectivity
   _build_face_dimension
   _build_face_edges_connectivity
   _populate_cartesian_xyz_coord
   _populate_lonlat_coord
=======
   grid.Grid.__init_ds_var_names__
   grid.Grid.__from_ds__
   grid.Grid.__from_vert__
   grid.Grid.__init_grid_var_attrs__
   grid.Grid._build_edge_node_connectivity
   grid.Grid._build_face_edges_connectivity
   grid.Grid._populate_cartesian_xyz_coord
   grid.Grid._populate_lonlat_coord
   grid.Grid._build_nNodes_per_face


Grid Helper Modules
--------------------
.. currentmodule:: uxarray
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   io._exodus._read_exodus
   io._exodus._encode_exodus
   io._exodus._get_element_type
   io._mpas._dual_to_ugrid
   io._mpas._primal_to_ugrid
   io._mpas._replace_padding
   io._mpas._replace_zeros
   io._mpas._to_zero_index
   io._mpas._set_global_attrs
   io._mpas._read_mpas
   io._ugrid._encode_ugrid
   io._ugrid._read_ugrid
   io._scrip._read_scrip
   io._scrip._encode_scrip
   io._scrip._to_ugrid
   utils.helpers._is_ugrid
   utils.helpers._replace_fill_values
