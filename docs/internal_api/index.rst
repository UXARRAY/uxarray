.. currentmodule:: uxarray

########
Internal API
########

This page shows already-implemented Uxarray internal API functions. You can also
check the draft `UXarray API
<https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
documentation to see the tentative whole API and let us know if you have any feedback!


UxDataset
=========
The ``uxarray.UxDataset`` class inherits from ``xarray.Dataset``. Below is a list of
features explicitly added to work on Unstructured Grids.


Class
-----
.. autosummary::
   :toctree: _autosummary
   UxDataset

Attributes
----------
.. autosummary::
   :toctree: _autosummary
   UxDataset._source_datasets
   UxDataset._uxgrid


Methods
-------
.. autosummary::
   :toctree: _autosummary
   __getitem__
   __setitem__
   _calculate_binary_op
   _construct_dataarray
   _construct_direct
   _copy
   _replace


UxDataArray
===========
The ``uxarray.UxDataArray`` class inherits from ``xarray.DataArray``. Below is a list of
features explicitly added to work on Unstructured Grids.

Class
-----
.. autosummary::
   :toctree: _autosummary
   UxDataArray


Attributes
----------
.. autosummary::
   :toctree: _autosummary
   UxDataArray._uxgrid


Methods
-------
.. autosummary::
   :toctree: _autosummary
   _construct_direct
   _copy
   _replace

Grid
===========

Class
----------
.. autosummary::
   :toctree: _autosummary
   Grid

Methods
-------
.. autosummary::
   :toctree: _autosummary
   __init_grid_var_names__
   __from_ds__
   __from_vert__
   __init_grid_var_attrs__

Attributes
----------
.. autosummary::
   :toctree: _autosummary
   Grid._Mesh2_node_z

Operators
---------
.. autosummary::
   :toctree: _autosummary
   Grid.__eq__
   Grid.__ne__

Helpers
===========

Connectivity
------------
.. autosummary::
   :toctree: _autosummary
   grid.connectivity._replace_fill_values
   grid.connectivity._build_nNodes_per_face
   grid.connectivity._build_edge_node_connectivity
   grid.connectivity._build_face_edges_connectivity

Coordinates
-----------
.. autosummary::
   :toctree: _autosummary
   grid.coordinates._populate_cartesian_xyz_coord
   grid.coordinates._populate_lonlat_coord

IO
---
.. autosummary::
   :toctree: _autosummary
   io.utils._parse_grid_type
