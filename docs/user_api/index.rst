.. currentmodule:: uxarray

########
User API
########

This page shows already-implemented Uxarray user API functions. You can also
check the draft `Uxarray API
<https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
documentation to see the tentative whole API and let us know if you have any feedback!


UxDataset
=========
The ``uxarray.UxDataset`` class inherits from ``xarray.Dataset``. Below is a list of explicitly added
or overloaded features guaranteed to work on Unstructured Grids.


Class
-----
.. autosummary::
   :toctree: _autosummary

   UxDataset

IO
----------
.. autosummary::
   :toctree: _autosummary

    core.api.open_dataset
    core.api.open_mfdataset

Attributes
----------
.. autosummary::
   :toctree: _autosummary

    UxDataset.uxgrid
    UxDataset.source_datasets

Methods
-------
.. autosummary::
   :toctree: _autosummary

    UxDataset.info
    UxDataset.integrate


UxDataArray
===========
The ``uxarray.UxDataArray`` class inherits from ``xarray.DataArray``. Below is a list of explicitly added
or overloaded features guaranteed to work on Unstructured Grids.

Class
-----
.. autosummary::
   :toctree: _autosummary

   UxDataArray


Attributes
----------
.. autosummary::
   :toctree: _autosummary

   UxDataArray.uxgrid


Methods
-------
.. autosummary::
   :toctree: _autosummary



Grid
===========

Class
----------
.. autosummary::
   :toctree: _autosummary

   Grid

IO
----------
.. autosummary::
   :toctree: _autosummary

   core.api.open_grid


Methods
-------
.. autosummary::
   :toctree: _autosummary

   Grid.calculate_total_face_area
   Grid.compute_face_areas
   Grid.encode_as
   Grid.integrate

Attributes
----------
.. autosummary::
   :toctree: _autosummary

Helpers
===========

Place into categories under utils eventually
----------------
.. currentmodule:: uxarray
.. autosummary::
   :toctree: _autosummary

   utils.helpers.calculate_face_area
   utils.helpers.calculate_spherical_triangle_jacobian
   utils.helpers.calculate_spherical_triangle_jacobian_barycentric
   utils.helpers.close_face_nodes
   utils.helpers.get_all_face_area_from_coords
   utils.helpers.get_gauss_quadratureDG
   utils.helpers.get_tri_quadratureDG
   utils.helpers.grid_center_lat_lon
   utils.helpers.node_xyz_to_lonlat_rad
   utils.helpers.node_lonlat_rad_to_xyz
   utils.helpers.normalize_in_place
   utils.helpers.parse_grid_type
