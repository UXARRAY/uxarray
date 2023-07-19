.. currentmodule:: uxarray

########
User API
########

This page shows already-implemented Uxarray user API functions. You can also
check the draft `UXarray API
<https://github.com/UXARRAY/uxarray/blob/main/docs/user_api/uxarray_api.md>`_
documentation to see the tentative whole API and let us know if you have any feedback!



UxDataset
=========
A ``xarray.Dataset``-like, multi-dimensional, in memory, array database.
Inherits from ``xarray.Dataset`` and has its own unstructured grid-aware
dataset operators and attributes through the ``uxgrid`` accessor.

Below is a list of features explicitly added to `UxDataset` to work on
Unstructured Grids:


Class
-----
.. autosummary::
   :toctree: _autosummary

   UxDataset

IO
----------
.. autosummary::
   :toctree: _autosummary

    open_dataset
    open_mfdataset

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
N-dimensional ``xarray.DataArray``-like array. Inherits from `xarray.DataArray`
and has its own unstructured grid-aware array operators and attributes through
the ``uxgrid`` accessor.

Below is a list of features explicitly added to `UxDataset` to work on
Unstructured Grids:

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



Grid
===========
Unstructured grid topology definition to store stores grid topology dimensions,
coordinates, variables and provides grid-specific functions.

Can be used standalone to explore an unstructured grid topology, or can be
seen as the property of ``uxarray.UxDataset`` and ``uxarray.DataArray`` to make
them unstructured grid-aware data sets and arrays.

Class
----------
.. autosummary::
   :toctree: _autosummary

   Grid

IO
----------
.. autosummary::
   :toctree: _autosummary

   open_grid


Methods
-------
.. autosummary::
   :toctree: _autosummary

   Grid.calculate_total_face_area
   Grid.compute_face_areas
   Grid.encode_as
   Grid.integrate
   Grid.copy

Attributes
----------
.. autosummary::
   :toctree: _autosummary

   Grid.Mesh2
   Grid.parsed_attrs
   Grid.nMesh2_node
   Grid.nMesh2_face
   Grid.nMesh2_edge
   Grid.nMaxMesh2_face_nodes
   Grid.nMaxMesh2_face_edges
   Grid.nNodes_per_face
   Grid.Mesh2_node_x
   Grid.Mesh2_node_y
   Grid.Mesh2_face_x
   Grid.Mesh2_face_y
   Grid.Mesh2_face_nodes
   Grid.Mesh2_edge_nodes
   Grid.Mesh2_face_edges


Helpers
===========

.. currentmodule:: uxarray
.. autosummary::
   :toctree: _autosummary

   calculate_face_area
   calculate_spherical_triangle_jacobian
   calculate_spherical_triangle_jacobian_barycentric
   close_face_nodes
   get_all_face_area_from_coords
   get_gauss_quadratureDG
   get_tri_quadratureDG
   grid_center_lat_lon
   node_xyz_to_lonlat_rad
   node_lonlat_rad_to_xyz
   normalize_in_place
   parse_grid_type
