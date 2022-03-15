.. currentmodule:: uxarray

Internal API
============

This page shows already-implemented Uxarray internal API functions. You can also
check the draft `Uxarray API
<https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
documentation to see the tentative whole API and let us know if you have any feedback!

Routines
--------

.. autosummary::
   :nosignatures:
   :toctree: ./generated/

   grid.Grid.__init__
   grid.Grid.__from_file__
   grid.Grid.__from_vert__
   _exodus._read_exodus
   _exodus._write_exodus
   _exodus._get_element_type
   _ugrid._write_ugrid
   _ugrid._read_ugrid
