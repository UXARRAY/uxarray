.. currentmodule:: uxarray

.. _algorithm-citations:

Algorithm-Level Citations
==========================

In addition to the package-level Zenodo citation, several spherical geometry and
regridding algorithms in UXarray implement methods from peer-reviewed publications.
Please cite the corresponding publication(s) below, in addition to the UXarray
software citation, if your work depends directly on the details of one of these
algorithms, for example describing the algorithm in detail or comparing it against
other algorithms. This does not apply to typical use of these APIs as part of an
analysis workflow, such as tutorials, internal tools, or research that simply
relies on UXarray's computed outputs.

The definitions and geometric conventions for nodes, edges, and faces used
throughout UXarray are based on:

    |cite-gmd|

Several of the intersection and geometry operators are additionally based on:

    |cite-siam|

.. |cite-gmd| replace:: Chen, H., Ullrich, P. A., Panetta, J., Marsico, D., Hanke, M., Jain, R.,
   Zhang, C., and Jacob, R. L. (2026). "Accurate and Robust Geometric Algorithms for
   Regridding on the Sphere." *Geoscientific Model Development*, 19(14), 6545-6570.
   `doi:10.5194/gmd-19-6545-2026 <https://doi.org/10.5194/gmd-19-6545-2026>`__
   (:download:`BibTeX <_static/citations/chen2026-gmd.bib>`)

.. |cite-siam| replace:: Chen, H., Ullrich, P. A., and Panetta, J. (2026). "Fast and Accurate
   Intersections on a Sphere." *SIAM Journal on Scientific Computing*, 48(2), B208-B232.
   `doi:10.1137/25M1737614 <https://doi.org/10.1137/25M1737614>`__
   (:download:`BibTeX <_static/citations/chen2026-siam.bib>`)

.. |cite-shewchuk| replace:: Shewchuk, J. R. (1997). "Adaptive Precision Floating-Point
   Arithmetic and Fast Robust Geometric Predicates." *Discrete & Computational Geometry*,
   18, 305-363. `doi:10.1007/PL00009321 <https://doi.org/10.1007/PL00009321>`__
   (:download:`BibTeX <_static/citations/shewchuk1997.bib>`)

.. |cite-knuth| replace:: Knuth, D. E. (1997). *The Art of Computer Programming, Volume 2:
   Seminumerical Algorithms* (3rd ed.). Addison-Wesley, Section 4.2.2, Theorem B.
   (:download:`BibTeX <_static/citations/knuth1997.bib>`)

.. |cite-dekker| replace:: Dekker, T. J. (1971). "A Floating-Point Technique for Extending
   the Available Precision." *Numerische Mathematik*, 18, 224-242.
   `doi:10.1007/BF01397083 <https://doi.org/10.1007/BF01397083>`__
   (:download:`BibTeX <_static/citations/dekker1971.bib>`)

.. |cite-higham| replace:: Higham, N. J. (2002). *Accuracy and Stability of Numerical
   Algorithms* (2nd ed.). Society for Industrial and Applied Mathematics.
   `doi:10.1137/1.9780898718027 <https://doi.org/10.1137/1.9780898718027>`__
   (:download:`BibTeX <_static/citations/higham2002.bib>`)

.. |cite-jeannerod| replace:: Jeannerod, C.-P., Louvet, N., and Muller, J.-M. (2013).
   "Further Analysis of Kahan's Algorithm for the Accurate Computation of 2 × 2
   Determinants." *Mathematics of Computation*, 82, 2245-2264.
   `doi:10.1090/S0025-5718-2013-02679-8 <https://doi.org/10.1090/S0025-5718-2013-02679-8>`__
   (:download:`BibTeX <_static/citations/jeannerod2013.bib>`)

.. |cite-rump| replace:: Rump, S. M. (2023). "Fast and Accurate Computation of the
   Euclidean Norm of a Vector." *Japan Journal of Industrial and Applied Mathematics*, 40.
   `doi:10.1007/s13160-023-00593-8 <https://doi.org/10.1007/s13160-023-00593-8>`__
   (:download:`BibTeX <_static/citations/rump2023.bib>`)

Algorithm-to-Publication Mapping
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Documentation section
     - API or implementation
     - Required citation(s)
   * - `Descriptors <api.html#descriptors>`__
     - :py:attr:`~uxarray.Grid.bounds`

       :py:attr:`~uxarray.Grid.face_bounds_lon`

       :py:attr:`~uxarray.Grid.face_bounds_lat`
     - |cite-gmd|
   * - `Zonal Average <api.html#zonal-average>`__
     - All zonal-average remapping implementations (e.g. :py:meth:`~uxarray.UxDataArray.zonal_mean`)
     - |cite-siam|
   * - `Spherical Geometry: Intersections <api.html#intersections>`__
     - All spherical-intersection APIs in this section (:py:func:`~uxarray.grid.intersections.gca_gca_intersection`, :py:func:`~uxarray.grid.intersections.gca_const_lat_intersection`, :py:func:`~uxarray.grid.intersections.get_number_of_intersections`)
     - **Cite both:**

       |cite-gmd|

       |cite-siam|
   * - `Spherical Geometry: Arcs <api.html#arcs>`__
     - :py:func:`~uxarray.grid.arcs.in_between`

       :py:func:`~uxarray.grid.arcs.point_within_gca`
     - No new citation required. Expected to be removed in a future release.
   * - `Spherical Geometry: Arcs <api.html#arcs>`__
     - :py:func:`~uxarray.grid.arcs.extreme_gca_latitude`
     - |cite-gmd|
   * - `Spherical Geometry: Arcs <api.html#arcs>`__
     - :py:func:`~uxarray.grid.arcs.orient3d_on_sphere`

       :py:func:`~uxarray.grid.arcs.on_minor_arc`
     - |cite-shewchuk|
   * - `Compensated Arithmetic <api.html#compensated-arithmetic>`__
     - :py:func:`~uxarray.utils.computing.two_sum`
     - |cite-knuth|
   * - `Compensated Arithmetic <api.html#compensated-arithmetic>`__
     - :py:func:`~uxarray.utils.computing.two_prod`
     - |cite-dekker|
   * - `Compensated Arithmetic <api.html#compensated-arithmetic>`__
     - :py:func:`~uxarray.utils.computing.diff_of_products`
     - **Cite both:**

       |cite-higham|

       |cite-jeannerod|
   * - `Compensated Arithmetic <api.html#compensated-arithmetic>`__
     - :py:func:`~uxarray.utils.computing.accucross`

       :py:func:`~uxarray.utils.computing.accucross_pair`
     - |cite-gmd|
   * - `Compensated Arithmetic <api.html#compensated-arithmetic>`__
     - :py:func:`~uxarray.utils.computing.acc_sqrt_re`
     - |cite-rump|
