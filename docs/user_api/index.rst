.. currentmodule:: uxarray

User API
========

This page shows already-implemented Uxarray user API functions. You can also
check the draft `Uxarray API
<https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
documentation to see the tentative whole API and let us know if you have any feedback!

Grid Class
----------
.. autosummary::
   :toctree: _autosummary

   grid.Grid


Grid Methods
------------
.. autosummary::
   :toctree: _autosummary

   grid.Grid.calculate_total_face_area
   grid.Grid.compute_face_areas
   grid.Grid.encode_as
   grid.Grid.integrate

Helper Functions
----------------
.. autosummary::
   :toctree: _autosummary

   helpers.calculate_face_area
   helpers.calculate_spherical_triangle_jacobian
   helpers.calculate_spherical_triangle_jacobian_barycentric
   helpers.get_all_face_area_from_coords
   helpers.get_gauss_quadratureDG
   helpers.get_tri_quadratureDG
   helpers.grid_center_lat_lon
   helpers.parse_grid_type
   helpers.node_xyz_to_lonlat_rad
   helpers.node_lonlat_rad_to_xyz
   helpers.normalize_in_place
   helpers.close_face_nodes

Multi-Precision Helper Functions
-------------------------------
.. autosummary::
   :toctree: _autosummary

   multi_precision_helpers.convert_to_multiprecision
   multi_precision_helpers.unique_coordinates_multiprecision
   multi_precision_helpers.decimal_digits_to_precision_bits
   multi_precision_helpers.precision_bits_to_decimal_digits
