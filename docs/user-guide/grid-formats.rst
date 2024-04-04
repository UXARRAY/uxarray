.. currentmodule:: uxarray


======================
Supported Grid Formats
======================

Overview
========

UXarray supports reading in multiple unstructured grid formats and encoding them in the UGRID conventions.

* UGRID
* MPAS
* SCRIP
* EXODUS
* ESMF

While each of these formats can be encoded in the UGRID conventions, the amount of information that is parsed from them
varies. The following sections describes how each format is converted into the UGRID conventions and what variables
are directly parsed.


Parsing Support
===============

.. note::

   While not all variables are present in each format, Uxarray provides functionality for deriving additional variables.
   More information can be found HERE

Coordinates
-----------

.. raw:: html

   <style>
   .yes-cell {
     background-color: green;
     color: white;
     text-align: center;
   }

   .no-cell {
     background-color: red;
     color: white;
     text-align: center;
   }

    /* Set column widths */
   th, td {
     width: 130px; /* Adjust this value as needed */
     text-align: center; /* Center align text */
     border: 2px solid black;
   }
   </style>

   <table border="1">
     <tr>
       <th></th>
       <th>UGRID</th>
       <th>MPAS</th>
       <th>SCRIP</th>
       <th>EXODUS</th>
       <th>ESMF</th>
     </tr>
     <tr>
       <td>node_latlon</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
     </tr>
     <tr>
       <td>edge_latlon</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
     </tr>
     <tr>
       <td>face_latlon</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
     </tr>
     <tr>
       <td>node_xyz</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
     </tr>
     <tr>
       <td>edge_xyz</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
     </tr>
     <tr>
       <td>face_xyz</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
     </tr>
   </table>


Connectivity
------------
.. raw:: html

   <style>
   .yes-cell {
     background-color: green;
     color: white;
     text-align: center;
   }

   .no-cell {
     background-color: red;
     color: white;
     text-align: center;
   }

    /* Set column widths */
   th, td {
     width: 140px; /* Adjust this value as needed */
     text-align: center; /* Center align text */
     border: 2px solid black;
   }
   </style>

   <table border="1">
     <tr>
       <th></th>
       <th>UGRID</th>
       <th>MPAS</th>
       <th>SCRIP</th>
       <th>EXODUS</th>
       <th>ESMF</th>
     </tr>
     <tr>
       <td>face_node</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
     </tr>
     <tr>
       <td>face_edge</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
     </tr>
     <tr>
       <td>face_face</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
     </tr>
     <tr>
       <td>edge_node</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
     </tr>
     <tr>
       <td>edge_edge</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
     </tr>
     <tr>
       <td>edge_face</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
     </tr>
     <tr>
       <td>node_node</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
     </tr>
     <tr>
       <td>node_edge</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
     </tr>
     <tr>
       <td>node_face</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
     </tr>
   </table>



UGRID
=====

Section TODO

MPAS
====

Section TODO

SCRIP
=====

Section TODO

EXODUS
======

Section TODO

ESMF
====

Section TODO
