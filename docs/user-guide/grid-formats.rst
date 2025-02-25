.. currentmodule:: uxarray


===============================
Supported Models & Grid Formats
===============================

Overview
========

UXarray is written primarily around the UGRID conventions, which is a standard for storing unstructured grid model
output in the NetCDF file format. While some models produce output in the UGRID conventions (e.g. FESOM2), many
models have their own grid format (e.g. MPAS, ICON). Below is a list of support grid formats and models.

* UGRID
* MPAS
* SCRIP
* EXODUS
* ESMF
* GEOS CS
* ICON
* FESOM2
* HEALPix

UGRID
=====

The UGRID conventions are a standard for for storing unstructured grid (a.k.a. unstructured mesh,
flexible mesh) model data in a Unidata Network Common Data Form (NetCDF) file.

These conventions are focussed on representing data for environmental applications, hence the motivation for
starting from the Climate & Forecasting (CF) Metadata Conventions,
The CF Conventions have been the standard in climate rx`earch for many years, and are being adopted by others as the metadata
standard (e.g. NASA, Open Geospatial Consortium). The CF conventions allow you to provide the geospatial and temporal coordinates
for scientific data, but currently assumes that the horizontal topology may be inferred from the i,j indices of structured
grids. The UGRID Conventions outline how to specify the topology of unstructured grids.

The standard was developed over a period of several years through the UGRID Google Group which had members from many
different unstructured grid modeling communities (including SELFE, ELCIRC, FVCOM, ADCIRC). From these discussions Bert
Jagers (Deltares) created the first draft of this document, and the community worked to develop version 1.0.

References
----------
* https://ugrid-conventions.github.io/ugrid-conventions/#ugrid-conventions-v10
* https://github.com/ugrid-conventions/ugrid-conventions
* https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#ugrid-conventions
* https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#mesh-topology-variables

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

References
----------
* https://mpas-dev.github.io/
* https://mpas-dev.github.io/files/documents/MPAS-MeshSpec.pdf

SCRIP
=====

The Spherical Coordinate Remapping and Interpolation Package (SCRIP) package is a software package used to generate
interpolation weights for remapping fields from one grid to another in spherical geometry.

A SCRIP format grid file is a NetCDF file for describing unstructured grids.

References
----------
* https://archive.org/details/manualzilla-id-6909486
* https://earthsystemmodeling.org/docs/release/ESMF_8_0_1/ESMF_refdoc/node3.html#SECTION03028100000000000000

EXODUS
======

EXODUS is a binary format based on NetCDF, leading to smaller file sizes compared to ASCII formats.
It is system independent and typically consists of nodes (geometric points), elements (e.g., triangles, tetrahedrons),
material properties, boundary conditions, and results from analysis.

Moreover, EXODUS facilitates efficient data storage and retrieval for computational simulations,
aiding in the management and analysis of complex engineering and scientific datasets.
It supports a wide range of finite element analysis applications and provides interoperability
with various simulation software packages. Additionally, the format ensures compatibility across different platforms,
enhancing collaboration and data exchange within the scientific community.

References
----------
* https://www.osti.gov/servlets/purl/10102115
* https://www.paraview.org/Wiki/ParaView/Users_Guide/Exodus_Reader

ESMF
====

The Earth System Modeling Framework (ESMF) is high-performance, flexible software infrastructure for building and
coupling weather, climate, and related Earth science applications. The ESMF defines an architecture for composing
complex, coupled modeling systems and includes data structures and utilities for developing individual models.

ESMF supports a custom unstructured grid file format for describing meshes, which is more compatible than the SCRIP
format.

References
----------
* https://earthsystemmodeling.org/about/
* https://earthsystemmodeling.org/docs/release/ESMF_8_0_1/ESMF_refdoc/node3.html#SECTION03028200000000000000

GEOS CS
=======

The Goddard Earth Observing System (GEOS) Cube Sphere (CS) grid format is equidistant gnomonic cubed-sphere grid with
6 identical faces that wrap a sphere, with some number of grid cells per face. For example, a C720 GEOS-CS grid has
6 faces, each with 720x720 elements.

References
----------
* https://gmao.gsfc.nasa.gov/gmaoftp/ops/GEOSIT_sample/doc/CS_Description_c180_v1.pdf


ICON
====
The climate model ICON is the central research tool at the Max Planck Institute for Meteorology (MPI-M). CON, which
obtains its name from the usage of spherical grids derived from the icosahedron (ICO) and the non-hydrostatic (N)
dynamics, originated as a joint project of the MPI-M and the German Weather Service (Deutscher Wetterdienst, DWD)
and has expanded to involve more development partners at the German Climate Computing Center (DKRZ),
the Swiss Federal Institute of Technology (ETH) in Zurich and the Karlsruhe Institute of Technology (KIT).
It includes component models for the atmosphere, the ocean and the land, as well as chemical and biogeochemical cycles,
all implemented on the basis of common data structures and sharing the same efficient technical infrastructure.
It is integrated and maintained by a group of experts for model development and application in the
institute’s Scientific Computing Laboratory.

References
----------
* https://mpimet.mpg.de/en/research/modeling
* https://scivis2017.dkrz.de/hd-cp-2/en-icon_grid.pdf

FESOM2
======
The Finite Volume Sea Ice-Ocean Model (FESOM2) is a Multi-resolution ocean general circulation
model that solves the equations of motion describing the ocean and sea ice using finite-volume methods
on unstructured computational grids. The model is developed and supported by researchers at the
Alfred Wegener Institute, Helmholtz Centre for Polar and Marine Research (AWI), in Bremerhaven, Germany.

References
----------
* https://fesom2.readthedocs.io/en/latest/index.html#

HEALPix
=======
The Hierarchical Equal Area isoLatitude Pixelisation (HEALPix) algorithm is a method for the pixelisation of the
2-sphere. It has three defining qualities.
- The sphere is hierarchically tessellated into curvilinear quadrilaterals
- Areas of all pixels at a given resolution are identical
- Pixels are distributed on lines of constant latitude

References
----------
* https://easy.gems.dkrz.de/Processing/healpix/index.html#hierarchical-healpix-output
* https://healpix.sourceforge.io/
* https://healpix.jpl.nasa.gov/
* https://iopscience.iop.org/article/10.1086/427976

Parsed Variables
================

Each unstructured grid formats varies in the amount of information contained about the grid. UXarray parses the
variables represented in each format and represents them in the UGRID conventions.

.. note::

   Even though each unstructured grid format has a varying number of support variables, UXarray provides
   support for constructing additional variables, which is discussed in the next sections.


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
       <th>GEOS-CS</th>
       <th>ICON</th>
     </tr>
     <tr>
       <td>node_latlon</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
     </tr>
     <tr>
       <td>edge_latlon</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>

     </tr>
     <tr>
       <td>face_latlon</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
     </tr>
     <tr>
       <td>node_xyz</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
     </tr>
     <tr>
       <td>edge_xyz</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
     </tr>
     <tr>
       <td>face_xyz</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
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
       <th>GEOS-CS</th>
       <th>ICON</th>
     </tr>
     <tr>
       <td>face_node</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
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
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
     </tr>
     <tr>
       <td>face_face</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
     </tr>
     <tr>
       <td>edge_node</td>
       <td class="yes-cell">Yes</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
     </tr>
     <tr>
       <td>edge_edge</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
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
       <td class="no-cell">No</td>
       <td class="yes-cell">Yes</td>
     </tr>
     <tr>
       <td>node_node</td>
       <td class="yes-cell">Yes</td>
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
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
       <td class="no-cell">No</td>
       <td class="no-cell">No</td>
     </tr>
   </table>
