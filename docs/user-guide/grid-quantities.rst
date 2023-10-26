.. currentmodule:: uxarray

===============
Grid Quantities
===============

Overview
--------

In addition to Coordinate and Connectivity Variables, ``UXarray`` also houses other variables
that describe quantities on an unstructured grid.





.. list-table::
   :widths: 25 75 25 100
   :header-rows: 1

   * - Variable
     - Grid Access
     - Dimension
     - Summary
   * - Face Area
     - ``Grid.face_area``
     - :math:`(n_{face}, )`
     - Node Indices that make up each face


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
   * - Face Area
     - No
     - No
     - No
     - No
     - No
