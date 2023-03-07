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

   __init_grid_var_names__
   __from_ds__
   __from_vert__
   __init_grid_var_attrs__
   _populate_cartesian_xyz_coord
   _populate_lonlat_coord


Grid Helper Modules
--------------------
.. currentmodule:: uxarray
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   _exodus._read_exodus
   _exodus._encode_exodus
   _exodus._get_element_type
   _ugrid._encode_ugrid
   _ugrid._read_ugrid
   _scrip._read_scrip
   _scrip._encode_scrip
   _scrip._to_ugrid
   helpers._is_ugrid
   helpers._convert_node_xyz_to_lonlat_rad
   helpers._convert_node_lonlat_rad_to_xyz
   helpers._normalize_in_place
