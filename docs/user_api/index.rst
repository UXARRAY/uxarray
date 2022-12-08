.. currentmodule:: uxarray

User API
========

This page shows already-implemented Uxarray user API functions. You can also
check the draft `Uxarray API
<https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
documentation to see the tentative whole API and let us know if you have any feedback!

GridAccessor Class
------------------
.. autosummary::
   :toctree: _autosummary

   grid.GridAccessor


Grid Methods
------------
.. autosummary::
   :toctree: _autosummary

   grid.GridAccessor.calculate_total_face_area
   grid.GridAccessor.compute_face_areas
   grid.GridAccessor.encode_as
   grid.GridAccessor.integrate


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
