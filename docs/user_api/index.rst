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

Grid functions
--------------
.. autosummary::
   :toctree: _autosummary

   grid.Grid.saveas_file
   grid.Grid.write
   _exodus.read_exodus
   _exodus.write_exodus
   _ugrid.write_ugrid
   _ugrid.read_ugrid

Helper functions
----------------
.. autosummary::
   :toctree: _autosummary

   helpers.determine_file_type
