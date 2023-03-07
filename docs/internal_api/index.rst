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

   io._exodus._read_exodus
   io._exodus._encode_exodus
   io._exodus._get_element_type
   io._ugrid._encode_ugrid
   io._ugrid._read_ugrid
   io._scrip._read_scrip
   io._scrip._encode_scrip
   io._scrip._to_ugrid
   utils.helpers._is_ugrid
   utils.helpers._convert_node_xyz_to_lonlat_rad
   utils.helpers._convert_node_lonlat_rad_to_xyz
   utils.helpers._normalize_in_place
