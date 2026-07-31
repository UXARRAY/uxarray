.. currentmodule:: uxarray

.. _citation:

How to Cite UXarray
===================

Cite UXarray using the following text:

**UXarray Organization. (Year).
UXarray (version \<version\>) [Software].
Project Raijin & Project SEATS. doi:10.5281/zenodo.<doi-part-per-version>.**

Update the year, UXarray version, and DOI part for UXarray version as appropriate.

Cite all versions? You can cite all versions by using the DOI `10.5281/zenodo.5655065
<https://doi.org/10.5281/zenodo.5655065>`_. This DOI represents all versions, and it will
always resolve to the latest one. `Read more <https://zenodo.org/help/versioning>`_.

However, if you are interested in citing a specific version of the package, go to `all versions
<https://zenodo.org/search?q=parent.id%3A5655065&f=allversions%3Atrue&l=list&p=1&s=10&sort=version>`_ ,
open the version you want to cite, and check out the "Details" section on the right
sidebar to find DOI, and use the <doi-part-per-version> from that.

For example:

**UXarray Organization. (2021).
UXarray (version 2025.06.0) [Software].
Project Raijin & Project SEATS. doi:10.5281/zenodo.15757812.**

.. _algorithm-citations:

Algorithm-Level Citations
==========================

In addition to the package-level Zenodo citation above, several of the spherical
geometry and regridding algorithms implemented in UXarray are associated with
peer-reviewed methodological publications. If your work makes use of the APIs
listed below, please also cite the corresponding publication(s) alongside the
UXarray software citation.

The definitions and geometric conventions for nodes, edges, and faces used
throughout UXarray are based on:

    Chen, H., Ullrich, P. A., Panetta, J., Marsico, D., Hanke, M., Jain, R.,
    Zhang, C., and Jacob, R. L. (2026). "Accurate and Robust Geometric
    Algorithms for Regridding on the Sphere." *Geoscientific Model
    Development*, 19(14), 6545-6570.
    `doi:10.5194/gmd-19-6545-2026 <https://doi.org/10.5194/gmd-19-6545-2026>`_

Several of the intersection and remapping algorithms are additionally based on:

    Chen, H., Ullrich, P. A., and Panetta, J. (2026). "Fast and Accurate
    Intersections on a Sphere." *SIAM Journal on Scientific Computing*,
    48(2), B208-B232.
    `doi:10.1137/25M1737614 <https://doi.org/10.1137/25M1737614>`_

Complete BibTeX entries for these and the supporting numerical-methods
references below are maintained in
`docs/references.bib <https://github.com/UXARRAY/uxarray/blob/main/docs/references.bib>`_.

Algorithm-to-Publication Mapping
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Documentation section
     - API or implementation
     - Required citation(s)
   * - Grid bounds
     - :py:attr:`~uxarray.Grid.bounds`
     - Chen et al. (2026), *GMD*
   * - Grid bounds
     - :py:attr:`~uxarray.Grid.face_bounds_lon`
     - Chen et al. (2026), *GMD*
   * - Grid bounds
     - :py:attr:`~uxarray.Grid.face_bounds_lat`
     - Chen et al. (2026), *GMD*
   * - `Zonal Average <api.html#remapping>`__
     - All zonal-average remapping implementations (e.g. :py:meth:`~uxarray.UxDataArray.zonal_average`)
     - Chen, Ullrich & Panetta (2026), *SIAM J. Sci. Comput.*
   * - `Spherical Geometry: Intersections <api.html#remapping>`__
     - All spherical-intersection APIs in this section (:py:func:`~uxarray.grid.intersections.gca_gca_intersection`, :py:func:`~uxarray.grid.intersections.gca_const_lat_intersection`, :py:func:`~uxarray.grid.intersections.get_number_of_intersections`)
     - Cite both: Chen et al. (2026), *GMD*; Chen, Ullrich & Panetta (2026), *SIAM J. Sci. Comput.*
   * - `Spherical Geometry: Arcs <api.html#remapping>`__
     - :py:func:`~uxarray.grid.arcs.extreme_gca_latitude`
     - Chen et al. (2026), *GMD*
   * - `Spherical Geometry: Arcs <api.html#remapping>`__
     - :py:func:`~uxarray.grid.arcs.orient3d_on_sphere`
     - Shewchuk (1997)
   * - `Spherical Geometry: Arcs <api.html#remapping>`__
     - :py:func:`~uxarray.grid.arcs.on_minor_arc`
     - Shewchuk (1997)
   * - `Spherical Geometry: Arcs <api.html#remapping>`__
     - :py:func:`~uxarray.grid.arcs.in_between`
     - No new citation required. Expected to be removed in a future release.
   * - `Spherical Geometry: Arcs <api.html#remapping>`__
     - :py:func:`~uxarray.grid.arcs.point_within_gca`
     - No new citation required. Expected to be removed in a future release.
   * - `Compensated Arithmetic <api.html#remapping>`__
     - :py:func:`~uxarray.utils.computing.two_sum`
     - Knuth (1997), *TAOCP Vol. 2*, Sec. 4.2.2, Theorem B
   * - `Compensated Arithmetic <api.html#remapping>`__
     - :py:func:`~uxarray.utils.computing.two_prod`
     - Dekker (1971)
   * - `Compensated Arithmetic <api.html#remapping>`__
     - :py:func:`~uxarray.utils.computing.diff_of_products`
     - Cite both: Higham (2002); Jeannerod, Louvet & Muller (2013)
   * - `Compensated Arithmetic <api.html#remapping>`__
     - :py:func:`~uxarray.utils.computing.accucross`
     - Chen et al. (2026), *GMD*
   * - `Compensated Arithmetic <api.html#remapping>`__
     - :py:func:`~uxarray.utils.computing.accucross_pair`
     - Chen et al. (2026), *GMD*
   * - `Compensated Arithmetic <api.html#remapping>`__
     - :py:func:`~uxarray.utils.computing.acc_sqrt_re`
     - Rump (2023)
