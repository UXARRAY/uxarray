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

   uxarray.core.grid.Grid


Grid Methods
------------
.. currentmodule:: uxarray.core.grid.Grid
.. autosummary::
   :toctree: _autosummary

   calculate_total_face_area
   compute_face_areas
   encode_as
   integrate


Helper Functions
----------------
.. currentmodule:: uxarray
.. autosummary::
   :toctree: _autosummary

   utils.helpers.calculate_face_area
   utils.helpers.calculate_spherical_triangle_jacobian
   utils.helpers.calculate_spherical_triangle_jacobian_barycentric
   utils.helpers.get_all_face_area_from_coords
   utils.helpers.get_gauss_quadratureDG
   utils.helpers.get_tri_quadratureDG
   utils.helpers.grid_center_lat_lon
   utils.helpers.parse_grid_type
