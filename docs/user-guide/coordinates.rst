.. currentmodule:: uxarray

===========
Coordinates
===========

Overview
--------

``UXarray`` ~~~ todo

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
     - No
     - Yes
     - Yes
     - Yes
     - Yes
   * - Node (Cartesian)
     - No
     - Yes
     - Yes
     - Yes
     - Yes
   * - Edge (Spherical)
     - No
     - Yes
     - Yes
     - Yes
     - Yes
   * - Edge (Cartesian)
     - No
     - Yes
     - Yes
     - Yes
     - Yes
   * - Face (Spherical)
     - No
     - Yes
     - Yes
     - Yes
     - Yes
   * - Face (Cartesian)
     - No
     - Yes
     - Yes
     - Yes
     - Yes