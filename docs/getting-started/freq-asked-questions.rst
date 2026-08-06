.. currentmodule:: uxarray

==========================
Frequently Asked Questions
==========================

I want to learn more about UXarray (or other geospatial topics), where should I start?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We have extensive notebooks, tutorials, and learning resources on our site Project Pythia:
`Project Pythia Cookbooks <https://cookbooks.projectpythia.org/>`_ have a variety of domain specific workflows to learn from.
If you want more UXarray specific information and examples, look at our `Unstructured Grid Visualization Cookbook <https://projectpythia.org/unstructured-grid-viz-cookbook/>`_
where we have:

- `Unstructured Grids Overview <https://projectpythia.org/unstructured-grid-viz-cookbook/notebooks/foundations/unstructured-grids/>`_
- `Plotting Libraries and Live Demos <https://projectpythia.org/unstructured-grid-viz-cookbook/notebooks/foundations/plotting-libs/>`_
- `Comparison to Xarray <https://projectpythia.org/unstructured-grid-viz-cookbook/notebooks/plotting-with-uxarray/compare-xarray/>`_

And other useful guides!

What foundational assumptions does UXarray make about grids and geometry?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UXarray assumes horizontally unstructured grids, consistent with the
`UGRID conventions' 2D flexible mesh topology <https://ugrid-conventions.github.io/ugrid-conventions/#2d-flexible-mesh-mixed-triangles-quadrilaterals-etc-topology>`_,
with the extra assumption that all grid faces/cells are convex (all angles less than 180 degrees).
UXarray supports extra dimensions such as elevation or time by treating them separately.
For example, grids with vertical levels are treated as per the
`UGRID conventions' 3D layered mesh topology <https://ugrid-conventions.github.io/ugrid-conventions/#3d-layered-mesh-topology>`_.
Fully 3D unstructured topology is not supported.

UXarray's geometry algorithms assume the grid lies on a spherical surface,
spanning either globally (the entire sphere) or regionally (only a portion of the sphere).
The sphere radius can be adjusted but a unit sphere is assumed by default.
Coordinates default to spherical (latitude/longitude), but Cartesian coordinates (x, y, z) are supported as well.

See also: `Supported Models & Grid Formats <../user-guide/grid-formats.rst>`_.


Other questions coming soon!
