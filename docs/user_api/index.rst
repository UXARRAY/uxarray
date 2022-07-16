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
--------------
.. autosummary::
   :toctree: _autosummary

   grid.Grid.write
   grid.Grid.calculate_total_face_area
   grid.Grid.integrate
   grid.Grid.get_face_areas

Helper Functions
----------------
.. autosummary::
   :toctree: _autosummary

   helpers.get_all_face_area_from_coords
   helpers.calculate_face_area
   helpers.parse_grid_type
