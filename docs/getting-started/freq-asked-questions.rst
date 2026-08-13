.. currentmodule:: uxarray

==========================
Frequently Asked Questions
==========================

I want to learn more about unstructured grids (or other topics of the Earth System Sciences), where should I start?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Check out:
`Project Pythia <https://cookbooks.projectpythia.org/>`_ , a home for Python-justified learning resources that are open-source, community-owned, geoscience-focused, and high-quality. For instance,
`Pythia Cookbooks <https://cookbooks.projectpythia.org/>`_ provide a variety of advanced, domain-specific workflows. In particular, the
`Unstructured Grid Visualization Cookbook <https://projectpythia.org/unstructured-grid-viz-cookbook/>`_ houses
`foundational information <https://projectpythia.org/unstructured-grid-viz-cookbook/notebooks/foundations/unstructured-grids/>`_ along with a comprehensive
showcase of workflows & techniques for working with Unstructured Grids.


Read through `UGRID Conventions <https://ugrid-conventions.github.io/ugrid-conventions/>`_, which is what UXarray is written around to represent several
different unstructured mesh types in a unified, `Climate and Forecast metadata convention (CF) <https://cfconventions.org/>`_-compliant format.


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
