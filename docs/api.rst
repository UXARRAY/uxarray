.. currentmodule:: uxarray

.. _api:


API reference
=============

This page provides an auto-generated summary of UXarray's API. For more details
and examples, refer to the relevant chapters in the main part of the
documentation.


Top Level Functions
-------------------

.. autosummary::
   :toctree: generated/

   open_grid
   open_dataset
   open_multigrid
   open_mfdataset
   concat

Tutorial
--------

.. autosummary::
   :toctree: generated/

   tutorial.available_datasets
   tutorial.describe_dataset
   tutorial.file_path
   tutorial.file_paths
   tutorial.open_grid
   tutorial.open_dataset
   tutorial.open_mfdataset

Grid
----

Constructors
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Grid
   Grid.from_dataset
   Grid.from_face_vertices
   Grid.from_file
   Grid.from_healpix
   Grid.from_points
   Grid.from_structured
   Grid.from_topology


Dual Mesh Construction
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Grid.get_dual


Indexing
~~~~~~~~
.. autosummary::
   :toctree: generated/

   Grid.get_edges_at_constant_latitude
   Grid.get_edges_at_constant_longitude
   Grid.get_faces_at_constant_latitude
   Grid.get_faces_at_constant_longitude
   Grid.get_faces_between_latitudes
   Grid.get_faces_between_longitudes
   Grid.get_faces_containing_point
   Grid.isel
   Grid.inverse_indices


Dimensions
~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Grid.dims
   Grid.sizes
   Grid.n_node
   Grid.n_edge
   Grid.n_face
   Grid.n_max_face_nodes
   Grid.n_max_face_edges
   Grid.n_max_face_faces
   Grid.n_max_edge_edges
   Grid.n_max_node_faces
   Grid.n_max_node_edges
   Grid.n_nodes_per_face


Spherical Coordinates
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Grid.coordinates
   Grid.node_lon
   Grid.node_lat
   Grid.edge_lon
   Grid.edge_lat
   Grid.face_lon
   Grid.face_lat


Cartesian Coordinates
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Grid.coordinates
   Grid.node_x
   Grid.node_y
   Grid.node_z
   Grid.edge_x
   Grid.edge_y
   Grid.edge_z
   Grid.face_x
   Grid.face_y
   Grid.face_z


Connectivity
~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   Grid.connectivity
   Grid.face_node_connectivity
   Grid.face_edge_connectivity
   Grid.face_face_connectivity
   Grid.edge_node_connectivity
   Grid.edge_edge_connectivity
   Grid.edge_face_connectivity
   Grid.node_node_connectivity
   Grid.node_edge_connectivity
   Grid.node_face_connectivity


Descriptors
~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   Grid.descriptors
   Grid.antimeridian_face_indices
   Grid.bounds
   Grid.boundary_node_indices
   Grid.boundary_edge_indices
   Grid.boundary_face_indices
   Grid.edge_node_distances
   Grid.edge_face_distances
   Grid.face_areas
   Grid.face_bounds_lon
   Grid.face_bounds_lat
   Grid.is_subset
   Grid.global_sphere_coverage
   Grid.max_face_radius
   Grid.partial_sphere_coverage
   Grid.sphere_radius
   Grid.triangular


Attributes
~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   Grid.attrs


Methods
~~~~~~~
.. autosummary::
   :toctree: generated/

   Grid.chunk
   Grid.copy
   Grid.calculate_total_face_area
   Grid.compute_face_areas
   Grid.construct_face_centers
   Grid.get_ball_tree
   Grid.get_kd_tree
   Grid.get_spatial_hash
   Grid.get_faces_containing_point
   Grid.normalize_cartesian_coordinates
   Grid.validate


Inheritance of Xarray Functionality
-----------------------------------

The primary data structures in UXarray, ``uxarray.UxDataArray`` and ``uxarray.UxDataset`` inherit from ``xarray.DataArray`` and
``xarray.Dataset`` respectively. This means that they contain the same methods and attributes that are present in Xarray, with
new additions and some overloaded as discussed in the next sections. For a detailed list of Xarray specific behavior
and functionality, please refer to Xarray's `documentation <https://docs.xarray.dev/en/stable/>`_.


UxDataArray
-----------

Constructors
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   UxDataArray
   UxDataArray.from_xarray
   UxDataArray.from_healpix


Dual Mesh Construction
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   UxDataArray.get_dual


Selection & Indexing
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   UxDataArray.isel
   UxDataArray.where


Grid Accessor
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   UxDataArray.uxgrid


UxDataset
-----------

Constructors
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   UxDataset
   UxDataset.from_dataframe
   UxDataset.from_dict
   UxDataset.from_healpix
   UxDataset.from_structured
   UxDataset.from_xarray


Dual Mesh Construction
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   UxDataset.get_dual


Grid Accessor
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   UxDataset.uxgrid


Attributes
~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   UxDataset.source_datasets


Methods
~~~~~~~
.. autosummary::
   :toctree: generated/

   UxDataset.info


Selection & Indexing
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   UxDataset.isel
   UxDataset.sel
   UxDataset.where


Conversion Methods
------------------

UXarray provides functionality to convert its unstructured grids representation to other data structures that can be ingested by existing, widely used tools, such as Matplotlib and Cartopy. This allows users to keep using their workflows with such tools.


Grid
~~~~

.. autosummary::
   :toctree: generated/

   Grid.to_geodataframe
   Grid.to_linecollection
   Grid.to_polycollection
   Grid.to_xarray


UxDataArray
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   UxDataArray.to_dataset
   UxDataArray.to_geodataframe
   UxDataArray.to_polycollection
   UxDataArray.to_raster
   UxDataArray.to_xarray


UxDataset
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   UxDataset.to_array
   UxDataset.to_xarray


Plotting
--------

UXarray's plotting API is written using `hvPlot <https://hvplot.holoviz.org/>`_. We also support standalone functions
for pure Matplotlib and Cartopy workflows.

.. seealso::

    `Plotting User Guide Section <https://uxarray.readthedocs.io/en/latest/user-guide/plotting.html>`_
    `Plotting with Matplotlib User Guide Section <https://uxarray.readthedocs.io/en/latest/user-guide/mpl.html>`_


Grid
~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Grid.plot
   Grid.plot.mesh
   Grid.plot.edges
   Grid.plot.node_coords
   Grid.plot.nodes
   Grid.plot.face_coords
   Grid.plot.face_centers
   Grid.plot.edge_coords
   Grid.plot.edge_centers
   Grid.plot.face_degree_distribution
   Grid.plot.face_area_distribution


UxDataArray
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   UxDataArray.plot
   UxDataArray.plot.polygons
   UxDataArray.plot.points
   UxDataArray.plot.line
   UxDataArray.plot.scatter


Subsetting
----------

.. seealso::

    `Subsetting User Guide Section <https://uxarray.readthedocs.io/en/latest/user-guide/subset.html>`_


Grid
~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Grid.subset
   Grid.subset.nearest_neighbor
   Grid.subset.bounding_box
   Grid.subset.bounding_circle
   Grid.subset.constant_latitude
   Grid.subset.constant_longitude
   Grid.subset.constant_latitude_interval
   Grid.subset.constant_longitude_interval


UxDataArray
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   UxDataArray.subset
   UxDataArray.subset.nearest_neighbor
   UxDataArray.subset.bounding_box
   UxDataArray.subset.bounding_circle
   UxDataArray.subset.constant_latitude
   UxDataArray.subset.constant_longitude
   UxDataArray.subset.constant_latitude_interval
   UxDataArray.subset.constant_longitude_interval


Cross Sections
--------------

.. seealso::

    `Cross Sections User Guide Section <https://uxarray.readthedocs.io/en/latest/user-guide/cross-sections.html>`_


.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   UxDataArray.cross_section


Remapping
---------

.. seealso::

    `Remapping User Guide Section <https://uxarray.readthedocs.io/en/latest/user-guide/remapping.html>`_


UxDataArray
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   UxDataArray.remap
   UxDataArray.remap.nearest_neighbor
   UxDataArray.remap.inverse_distance_weighted
   UxDataArray.remap.bilinear


UxDataset
~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   UxDataset.remap
   UxDataset.remap.nearest_neighbor
   UxDataset.remap.inverse_distance_weighted
   UxDataset.remap.bilinear


Mathematical Operators
----------------------

.. autosummary::
   :toctree: generated/

   UxDataArray.curl
   UxDataArray.difference
   UxDataArray.divergence
   UxDataArray.gradient
   UxDataArray.integrate
   UxDataArray.scalardotgradient


Aggregations
------------

Topological
~~~~~~~~~~~

Topological aggregations apply an aggregation (i.e. averaging) on a per-element basis. For example, instead of computing
the average across all values, we can compute the average of all the nodes that surround each face and store the result
on each face.

.. seealso::

    `Topological Aggregations User Guide Section <https://uxarray.readthedocs.io/en/latest/user-guide/topological-aggregations.html>`_

.. autosummary::
   :toctree: generated/

   UxDataArray.topological_mean
   UxDataArray.topological_min
   UxDataArray.topological_max
   UxDataArray.topological_median
   UxDataArray.topological_std
   UxDataArray.topological_var
   UxDataArray.topological_sum
   UxDataArray.topological_prod
   UxDataArray.topological_all
   UxDataArray.topological_any


Azimuthal
~~~~~~~~~

Azimuthal aggregations apply an aggregation (i.e. averaging) along circles of constant great-circle distance from a specified point on the sphere.

.. autosummary::
   :toctree: generated/

   UxDataArray.azimuthal_mean


Zonal Average
~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   UxDataArray.zonal_average
   UxDataArray.zonal_mean


Weighted
~~~~~~~~
.. autosummary::
   :toctree: generated/

   UxDataArray.weighted_mean


Spherical Geometry
------------------

Intersections
~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   grid.intersections.gca_gca_intersection
   grid.intersections.gca_const_lat_intersection


Arcs
~~~~

.. autosummary::
   :toctree: generated/

   grid.arcs.in_between
   grid.arcs.point_within_gca
   grid.arcs.extreme_gca_latitude


Accurate Computing
------------------

.. autosummary::
   :toctree: generated/

   utils.computing.cross_fma
   utils.computing.dot_fma
