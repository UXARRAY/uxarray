.. currentmodule:: uxarray

==========================
Frequently Asked Questions
==========================

What does UXarray assume about grids?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UXarray assumes that each grid spans some or all of a 2D spherical surface, and contains only convex faces.
While other dimensions such as time or elevation can be represented easily enough,
the unstructured part of the grid cannot vary across those dimensions.
UXarray also assumes that the grid can be represented using UGRID conventions,
even if it isn't input directly in UGRID format.
See also: `Supported Models & Grid Formats <user-guide/grid-formats.rst>`_.


Other questions coming soon!
