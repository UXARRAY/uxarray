.. currentmodule:: uxarray


======================
Supported Grid Formats
======================

Overview
========

UXarray is written around the UGRID conventions, which is a standard for storing unstructured grid model data in the
NetCDF file format. All internal grid


* UGRID
* MPAS
* SCRIP
* EXODUS
* ESMF

While each of these formats can be encoded in the UGRID conventions, the amount of information that is parsed from them
varies. The following sections describes how each format is converted into the UGRID conventions and what variables
are directly parsed.


UGRID
=====

The UGRID conventions are a standard for for storing unstructured grid (a.k.a. unstructured mesh,
flexible mesh) model data in a Unidata Network Common Data Form (NetCDF) file.

These conventions are focussed on representing data for environmental applications, hence the motivation for
starting from the Climate & Forecasting (CF) Metadata Conventions,
The CF Conventions have been the standard in climate research for many years, and are being adopted by others as the metadata
standard (e.g. NASA, Open Geospatial Consortium). The CF conventions allow you to provide the geospatial and temporal coordinates
for scientific data, but currently assumes that the horizontal topology may be inferred from the i,j indices of structured
grids. The UGRID Conventions outline how to specify the topology of unstructured grids.

The standard was developed over a period of several years through the UGRID Google Group which had members from many
different unstructured grid modeling communities (including SELFE, ELCIRC, FVCOM, ADCIRC). From these discussions Bert
Jagers (Deltares) created the first draft of this document, and the community worked to develop version 1.0.

https://ugrid-conventions.github.io/ugrid-conventions/#ugrid-conventions-v10

https://github.com/ugrid-conventions/ugrid-conventions

https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#ugrid-conventions

https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#mesh-topology-variables

MPAS
====

The Model for Prediction Across Scales (MPAS) is a collaborative project for developing atmosphere, ocean and other
earth-system simulation components for use in climate, regional climate and weather studies.

The defining features of MPAS are the unstructured Voronoi meshes and C-grid discretization used as the basis for many of
the model components.  The unstructured Voronoi meshes, formally Spherical Centriodal Voronoi Tesselations (SCVTs), allow
for both quasi-uniform discretization of the sphere and local refinement.  The C-grid discretization, where the normal
component of velocity on cell edges is prognosed, is especially well-suited for higher-resolution, mesoscale atmosphere
and ocean simulations. The land ice model takes advantage of the SCVT-dual mesh, which is a triangular Delaunay
tessellation appropriate for use with Finite-Element-based discretizations.

https://mpas-dev.github.io/

https://mpas-dev.github.io/files/documents/MPAS-MeshSpec.pdf

SCRIP
=====

The Spherical Coordinate Remapping and Interpolation Package (SCRIP) package is a software package used to generate
interpolation weights for remapping fields from one grid to another in spherical geometry.

A SCRIP format grid file is a NetCDF file for describing unstructured grids.

https://archive.org/details/manualzilla-id-6909486

https://earthsystemmodeling.org/docs/release/ESMF_8_0_1/ESMF_refdoc/node3.html#SECTION03028100000000000000

EXODUS
======

Section TODO

ESMF
====

The Earth System Modeling Framework (ESMF) is high-performance, flexible software infrastructure for building and
coupling weather, climate, and related Earth science applications. The ESMF defines an architecture for composing
complex, coupled modeling systems and includes data structures and utilities for developing individual models.

ESMF supports a custom unstructured grid file format for describing meshes, which is more compatible than the SCRIP
format.

https://earthsystemmodeling.org/about/

https://earthsystemmodeling.org/docs/release/ESMF_8_0_1/ESMF_refdoc/node3.html#SECTION03028200000000000000

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
