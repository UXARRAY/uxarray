.. currentmodule:: uxarray



=================================
Grid File Formats and Differences
=================================

.. list-table:: Comparison Grids Supported by UXARRAY
   :header-rows: 1

   * - Feature
     - MPAS
     - Exodus
     - UGRID
     - SCRIP
   * - File Format
     - NetCDF
     - NetCDF
     - NetCDF
     - NetCDF
   * - Grid Type
     - Unstructured
     - Unstructured
     - Unstructured
     - Unstructured
   * - Topology
     - Nodes, Cells, Faces. (Can read primal and dual meshes)
     - Nodes, Elements
     - Nodes, Edges, Faces
     - Nodes, Elements
   * - Coordinates
     - Cartesian: xVertex, yVertex, zVertex. Spherical: lonVertex, latVertex.
     - Cartesian: coordx, coordy, coordz
     - Spherical: node_lon, node_lat
     - Spherical: grid_corner_lon, grid_corner_lat
   * - Connectivity
     - verticesOnEdge, verticesOnCell, edgesOnVertex, edgesOnCell, cellsOnVertex
     - connect
     - face_node_connectivity
     - grid_center_lon, grid_center_lat
   * - Attributes
     - dvEdge, dcEdge
     - api_version, floating_point_word_size, file_size
     - cf_role, topology_dimension
     -
   * - Cell Type
     - Arbitrary shaped 2D polygons
     - Arbitrary shaped 2D polygons
     - Arbitrary shaped 2D polygons
     - Arbitrary shaped 2D polygons
   * - File Extension
     - \*.nc
     - \*.exo, \*.e
     - \*.nc
     - \*.scrip, \*.nc


================================
Specific Details of Grid Formats
================================

UGRID
-----

MPAS
----

EXODUS
------

SCRIP
-----
