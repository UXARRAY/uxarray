.. currentmodule:: uxarray

===========
Conventions
===========

UGRID Conventions
==================

The UGRID conventions provide a standard for storing unstructured grid model data in the form of a NetCDF file.
UXarray uses the UGRID conventions as a foundation for representing unstructured grids.

.. admonition:: More Info
   :class: tip

   For further information about the UGRID conventions, please refer to the `official documentation`_

.. _official documentation: https://ugrid-conventions.github.io/ugrid-conventions/

Elements & Dimensions
=====================

An unstructured grid is composed of nodes, edges, and faces which either fully or partially cover some surface. In the
context of climate modelling, this surface is the surface of the Earth, typically represented as a sphere.

For example, output from a global atmospheric model covers the entire sphere, while a global ocean model would
not have any elements over land.



Nodes
-----
An unstructured grid contains ``n_node`` corner nodes, which define the corners of each face. It may also
contain `n_face` centroid nodes, which represent the center of each face, and ``n_edge``
edge nodes, which represent the center of each edge.

Edges
-----

An unstructured grid contains ``n_edge`` edges, which each connect two corner nodes to form an arc.

Faces
-----
An unstructured grid contains ``n_face`` `faces.

UXarray is built to support 2D flexible grids, meaning that each face can have a variable number of nodes surrounding
it.

Each face can have an independent number of nodes that surround it, which is represented through the
descriptor variable ``n_nodes_per_face``, which itself has a dimension of ``n_face`` The minimum
number of nodes per face is 3 (a triangle), with the maximum number being represented by the dimension
``n_max_face_nodes``

Coordinates
===========

Definitions
-----------

Spherical Coordinates
---------------------

.. list-table::
   :header-rows: 1

   * - Coordinate
     - Grid Attribute
     - Dimensions
     - Summary
   * - Node Longitude
     - ``Grid.node_lon``
     - ``(n_node,)``
     - Longitude of each corner node
   * - Node Latitude
     - ``Grid.node_lat``
     - ``(n_node,)``
     - Latitude of each corner node in degrees
   * - Edge Longitude
     - ``Grid.edge_lon``
     - ``(n_edge,)``
     - Longitude of the center of each edge
   * - Edge Latitude
     - ``Grid.edge_lat``
     - ``(n_edge,)``
     - Latitude of the center of each edge
   * - Face Longitude
     - ``Grid.face_lon``
     - ``(n_face,)``
     - Longitude of the center of each face
   * - Face Latitude
     - ``Grid.face_lat``
     - ``(n_face,)``
     - Latitude of the center of each face


.. note::

    All spherical coordinates are represented in degrees, with longitudes between (-180°, 180°) and latitudes between (-90°, 90°).


Cartesian Coordinates
---------------------

.. list-table::
   :header-rows: 1

   * - Coordinate
     - Grid Attribute
     - Dimensions
     - Summary
   * - Node X
     - ``Grid.node_x``
     - ``(n_node,)``
     - X location of each corner node
   * - Node Y
     - ``Grid.node_y``
     - ``(n_node,)``
     - Y location of each corner node
   * - Node Z
     - ``Grid.node_z``
     - ``(n_node,)``
     - Z location of each corner node
   * - Edge X
     - ``Grid.edge_x``
     - ``(n_edge,)``
     - X location of the center of each edge
   * - Edge Y
     - ``Grid.edge_y``
     - ``(n_edge,)``
     - Y location of the center of each edge
   * - Edge Z
     - ``Grid.edge_z``
     - ``(n_edge,)``
     - Z location of the center of each edge
   * - Face X
     - ``Grid.face_x``
     - ``(n_face,)``
     - X location of the center of each face
   * - Face Y
     - ``Grid.face_y``
     - ``(n_face,)``
     - Y location of the center of each face
   * - Face Z
     - ``Grid.face_z``
     - ``(n_face,)``
     - Z location of the center of each face

.. note::

    All Cartesian coordinates are represented in meters.


Connectivity
============
UXarray relies on connectivity variables to describe how various elements (i.e nodes, edges, faces) can be connected.

.. list-table::
   :widths: 15 30 35 30
   :header-rows: 1

   * - Connectivity
     - Grid Attribute
     - Dimensions
     - Summary
   * - Face Node
     - ``Grid.face_node_connectivity``
     - ``(n_face, n_max_face_nodes)``
     - Indices of the nodes that make up each face
   * - Face Edge
     - ``Grid.face_edge_connectivity``
     - ``(n_face, n_max_face_edges)``
     - Indices of the edges that surround each face
   * - Face Face
     - ``Grid.face_face_connectivity``
     - ``(n_face, n_max_face_faces)``
     - Indices of the faces that surround each face
   * - Edge Node
     - ``Grid.edge_node_connectivity``
     - ``(n_edge, 2)``
     - Indices of the two nodes that make up each edge
   * - Edge Edge
     - ``Grid.edge_edge_connectivity``
     - ``(n_edge, n_max_edge_edges)``
     - Indices of the edges that surround each edge
   * - Edge Face
     - ``Grid.edge_face_connectivity``
     - ``(n_edge, n_max_edge_faces)``
     - Indices of the faces that saddle each edge
   * - Node Edge
     - ``Grid.node_edge_connectivity``
     - ``(n_node, n_max_node_edges)``
     - Indices of the edges that surround each node
   * - Node Face
     - ``Grid.node_face_connectivity``
     - ``(n_node, n_max_node_faces)``
     - Indices of the faces that surround each node
