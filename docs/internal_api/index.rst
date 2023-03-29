.. currentmodule:: uxarray

Internal API
============

This page shows already-implemented Uxarray internal API functions. You can also
check the draft `Uxarray API
<https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
documentation to see the tentative whole API and let us know if you have any feedback!

Grid Methods
------------

.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   grid.Grid.__init_ds_var_names__
   grid.Grid.__from_ds__
   grid.Grid.__from_vert__
   grid.Grid.__init_grid_var_attrs__
   grid.Grid._populate_cartesian_xyz_coord
   grid.Grid._populate_lonlat_coord


Grid Helper Modules
--------------------
.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   _exodus._read_exodus
   _exodus._encode_exodus
   _exodus._get_element_type
   _ugrid._encode_ugrid
   _ugrid._read_ugrid
   _mpas._dual_to_ugrid
   _mpas._primal_to_ugrid
   _mpas._replace_padding
   _mpas._replace_zeros
   _mpas._to_zero_index
   _mpas._set_global_attrs
   _mpas._read_mpas
   _scrip._read_scrip
   _scrip._encode_scrip
   _scrip._to_ugrid
   helpers._is_ugrid
   helpers._convert_node_xyz_to_lonlat_rad
   helpers._convert_node_lonlat_rad_to_xyz
   helpers._normalize_in_place
   helpers._replace_fill_values
