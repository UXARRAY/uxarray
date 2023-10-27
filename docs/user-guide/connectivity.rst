.. currentmodule:: uxarray

============
Connectivity
============

``UXarray`` relies on connectivity arrays to describe how various elements (i.e nodes, edges, faces) related to each-other
geometrically.

.. list-table::
   :widths: 25 75 25 100
   :header-rows: 1

   * - Connectivity
     - Grid Access
     - Dimension
     - Summary
   * - Face Node
     - ``Grid.face_node_connectivity``
     - :math:`(n_{face}, n_{maxnodes})`
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


Connectivity Visuals
--------------------
The following visuals

--------------------
Face Node
--------------------
Add Visual Here
