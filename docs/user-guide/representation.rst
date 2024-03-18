.. currentmodule:: uxarray

=================
Conventions
=================

UGRID Conventions
==================

UXarray is heavily based off the UGRID conventions, using them as a foundation for representing unstructured grids.
The UGRID conventions provide a standard for storing unstructured grid model data the form of a NetCDF file.

Dimensions
==========

In UXarray, an unstructured grid is composed of faces that either fully or partially cover the surface of a
sphere (i.e. Earth in climate models). Each face is made up of Nodes and Edges.

Nodes
-----
An unstructured grid contains :math:`(n_{node})` corner nodes, which define the corners of each face. It may also
contain :math:`(n_{face})` centroid nodes, which represent the center of each face, and :math:`(n_{edge})`
edge nodes, which represent the center of each edge.

Edges
-----

An unstructured grid contains :math:`(n_{edge})` edges, which each connect two corner nodes to form an arc.

Faces
-----
An unstructured grid contains :math:`(n_{face})` faces.

UXarray is built to support 2D flexible grids, meaning that each face can have a variable number of nodes surrounding
it.

Each face can have an independent number of nodes that surround it, which is represented through the
descriptor variable ``n_nodes_per_face``, which itself has a dimension of :math:`(n_{face})`. The minimum
number of nodes per face is 3 (a triangle), with the maximum number being represented by the dimension
:math:`(n_{maxfacenodes})`

Coordinates
===========

Definitions
-----------

Spherical Coordinates
---------------------

.. list-table::
   :widths: 75 75 25 100
   :header-rows: 1

   * - Coordinate
     - Grid Access
     - Dimension
     - Summary
   * - Node Longitude
     - ``Grid.node_lon``
     - :math:`(n_{node},)`
     - Insert summary of coordinate here
   * - Node Latitude
     - ``Grid.node_lat``
     - :math:`(n_{node},)`
     - TODO
   * - Edge Longitude
     - ``Grid.edge_lon``
     - :math:`(n_{edge},)`
     - TODO
   * - Edge Latitude
     - ``Grid.edge_lat``
     - :math:`(n_{edge},)`
     - TODO
   * - Face Longitude
     - ``Grid.face_lon``
     - :math:`(n_{face},)`
     - TODO
   * - Face Latitude
     - ``Grid.face_lat``
     - :math:`(n_{face},)`
     - TODO


Cartesian Coordinates
---------------------

.. list-table::
   :widths: 50 75 25 100
   :header-rows: 1

   * - Coordinate
     - Grid Access
     - Dimension
     - Summary
   * - Node X
     - ``Grid.node_x``
     - :math:`(n_{node},)`
     - Insert summary of coordinate here
   * - Node Y
     - ``Grid.node_y``
     - :math:`(n_{node},)`
     - Insert summary of coordinate here
   * - Node Z
     - ``Grid.node_z``
     - :math:`(n_{node},)`
     - Insert summary of coordinate here
   * - Edge X
     - ``Grid.edge_x``
     - :math:`(n_{edge},)`
     - Insert summary of coordinate here
   * - Edge Y
     - ``Grid.edge_y``
     - :math:`(n_{edge},)`
     - Insert summary of coordinate here
   * - Edge Z
     - ``Grid.edge_z``
     - :math:`(n_{edge},)`
     - Insert summary of coordinate here
   * - Face X
     - ``Grid.face_x``
     - :math:`(n_{face},)`
     - Insert summary of coordinate here
   * - Face Y
     - ``Grid.face_y``
     - :math:`(n_{face},)`
     - Insert summary of coordinate here
   * - Face Z
     - ``Grid.face_z``
     - :math:`(n_{face},)`
     - Insert summary of coordinate here

Parsing & Construction Support
------------------------------
Below


.. list-table::
   :widths: 25 25 25 25 25 25
   :header-rows: 1

   * - Coordiniate
     - Construction
     - UGRID
     - MPAS
     - EXODUS
     - SCRIP
   * - Node (Spherical)
     - No *
     - Yes
     - Yes
     - Yes
     - Yes
   * - Node (Cartesian)
     - No *
     - Yes
     - Yes
     - Yes
     - Yes
   * - Edge (Spherical)
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - Edge (Cartesian)
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - Face (Spherical)
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - Face (Cartesian)
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes


Connectivity
============
UXarray relies on connectivity variables to describe how various elements (i.e nodes, edges, faces) can be connected.

.. list-table::
   :widths: 25 75 25 100
   :header-rows: 1

   * - Connectivity
     - Grid Access
     - Dimension
     - Summary
   * - Face Node
     - ``Grid.face_node_connectivity``
     - :math:`(n_{face}, n_{max\_face\_nodes})`
     - Node Indices that make up each face
   * - Edge Node
     - ``Grid.edge_node_connectivity``
     - :math:`(n_{edge}, 2)`
     - Node Indices that make up each edge
   * - Face Edge
     - ``Grid.face_edge_connectivity``
     - :math:`(n_{face}, n_{maxedges})`
     - Edge Indices that make up each face
   * - Node Edge
     - ``Grid.node_edge_connectivity``
     - :math:`(n_{edge}, 2)`
     - TODO
   * - Face Face
     - ``Grid.face_face_connectivity``
     - :math:`(n_{edge}, 2)`
     - Face Indices that saddle a given edge
   * - Edge Face
     - ``Grid.edge_face_connectivity``
     - :math:`(n_{edge}, 2)`
     - Face Indices that saddle a given edge
   * - Node Face
     - ``Grid.node_face_connectivity``
     - :math:`(n_{node}, 2)`
     - TODO

Parsing & Construction Support
------------------------------

Below


.. list-table::
   :widths: 25 25 25 25 25 25
   :header-rows: 1

   * - Connectivity
     - Construction
     - UGRID
     - MPAS
     - EXODUS
     - SCRIP
   * - Face Node
     - No *
     - Yes
     - Yes
     - Yes
     - Yes
   * - Edge Node
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - Face Edge
     - Yes
     - No
     - No
     - No
     - No
   * - Node Edge
     - Yes
     - No
     - No
     - No
     - No
   * - Face Face
     - Yes
     - No
     - No
     - No
     - No
   * - Edge Face
     - Yes
     - No
     - No
     - No
     - No
   * - Node Face
     - Yes
     - No
     - No
     - No
     - No
