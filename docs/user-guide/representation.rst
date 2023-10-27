.. currentmodule:: uxarray

===================
Grid Representation
===================

Geometric Elements
==================

Dimensions
==========

In ``UXarray``, an unstructured grid is defined as a set of Faces that either
fully or partially cover the surface of a sphere. Each face is composed of Nodes that make
up each corner, and Edges that connect each node to form a closed Face. All nodes, edges, and faces are unique
(i.e. no duplicates are stored).

Nodes
-----
An unstructured grid contains :math:`(n_{node})` corner nodes, which define the corners of each face. It may also
contain :math:`(n_{face})` centroid nodes, which represent the center of each face, and :math:`(n_{edge})`
edge nodes, which represent the center of each edge.

Edges
-----

An unstructured grid contains :math:`(n_{edge})` edges, which each connect two corner nodes.

Faces
-----
An unstructured grid contains :math:`(n_{face})` faces.

Each face can have an independent number of nodes that surround it, which is represented through the
descriptor variable ``n_nodes_per_face``, which itself has a dimension of :math:`(n_{face})`. The minimum
number of nodes per face is 3 (a triangle), with the maximum number being represented by the dimension
:math:`(n_{maxfacenodes})`






Data Mapping
============
