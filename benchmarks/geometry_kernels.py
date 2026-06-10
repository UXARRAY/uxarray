"""Micro-benchmarks for the EFT-based spherical geometry kernels.

These benchmarks target the individual functions introduced in the AccuSphGeom
port (PR #1513) so that the per-kernel overhead of compensated arithmetic can
be measured independently of higher-level UXarray operations.

All Numba functions are warmed (compiled) during ``setup`` so that benchmark
timings reflect steady-state throughput, not JIT compilation.
"""

import numpy as np

from uxarray.grid.arcs import on_minor_arc, orient3d_on_sphere
from uxarray.grid.intersections import (
    _accux_constlat,
    _accux_gca,
    _try_gca_const_lat_intersection,
    _try_gca_gca_intersection,
    gca_const_lat_intersection,
    gca_gca_intersection,
)
from uxarray.grid.point_in_face import _point_in_polygon_sphere
from uxarray.utils.computing import (
    acc_sqrt_re,
    accucross,
    accucross_pair,
    diff_of_products,
    two_prod,
    two_sum,
)


def _unit(v):
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Representative inputs — chosen to exercise the near-tangent regime
# ---------------------------------------------------------------------------

# Two arcs that intersect at a small angle (near-tangent, stress-tests EFT)
_W0 = _unit(np.array([1.0, 0.0, 0.1]))
_W1 = _unit(np.array([0.0, 1.0, 0.1]))
_V0 = _unit(np.array([0.5, -0.1, 0.8]))
_V1 = _unit(np.array([0.5, 0.9, 0.05]))

# Arc for const-lat test
_X1 = _unit(np.array([1.0, 0.0, 0.3]))
_X2 = _unit(np.array([0.0, 1.0, 0.3]))
_CONST_Z = 0.3

# Polygon for point-in-polygon (spherical triangle)
_POLY = np.array(
    [
        _unit(np.array([1.0, 0.0, 0.1])),
        _unit(np.array([0.0, 1.0, 0.1])),
        _unit(np.array([-1.0, 0.0, 0.5])),
    ],
    dtype=np.float64,
)
_Q_INSIDE = _unit(np.array([0.1, 0.3, 0.9]))
_Q_OUTSIDE = _unit(np.array([-0.5, -0.5, -0.7]))


class EFTPrimitives:
    """Benchmark the low-level EFT building blocks: two_sum, two_prod,
    diff_of_products, and acc_sqrt_re."""

    def setup(self):
        # Warm Numba
        two_sum(1.0, 1e-16)
        two_prod(1.23456789, 9.87654321)
        diff_of_products(1.0, 2.0, 3.0, 4.0)
        acc_sqrt_re(1.0 - 1e-15)

    def time_two_sum(self):
        two_sum(1.23456789012345678, 9.87654321098765432e-16)

    def time_two_prod(self):
        two_prod(1.23456789012345678, 9.87654321098765432)

    def time_diff_of_products(self):
        diff_of_products(1.23456789, 9.87654321, 1.23456788, 9.87654322)

    def time_acc_sqrt_re(self):
        acc_sqrt_re(1.0 - 1e-15)


class AccucrossKernels:
    """Benchmark the compensated cross-product kernels."""

    def setup(self):
        accucross(
            _W0[0], _W0[1], _W0[2],
            _W1[0], _W1[1], _W1[2],
        )
        accucross_pair(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        )

    def time_accucross(self):
        accucross(
            _W0[0], _W0[1], _W0[2],
            _W1[0], _W1[1], _W1[2],
        )

    def time_accucross_pair(self):
        n1x_hi, n1y_hi, n1z_hi, n1x_lo, n1y_lo, n1z_lo = accucross(
            _W0[0], _W0[1], _W0[2], _W1[0], _W1[1], _W1[2]
        )
        n2x_hi, n2y_hi, n2z_hi, n2x_lo, n2y_lo, n2z_lo = accucross(
            _V0[0], _V0[1], _V0[2], _V1[0], _V1[1], _V1[2]
        )
        accucross_pair(
            n1x_hi, n1y_hi, n1z_hi, n1x_lo, n1y_lo, n1z_lo,
            n2x_hi, n2y_hi, n2z_hi, n2x_lo, n2y_lo, n2z_lo,
        )


class OrientPredicates:
    """Benchmark the orient3d and on_minor_arc predicates."""

    def setup(self):
        orient3d_on_sphere(_W0, _W1, _V0)
        on_minor_arc(_V0, _W0, _W1)

    def time_orient3d_on_sphere(self):
        orient3d_on_sphere(_W0, _W1, _V0)

    def time_on_minor_arc(self):
        on_minor_arc(_V0, _W0, _W1)


class GCAGCAIntersection:
    """Benchmark all three layers of the GCA-GCA intersection stack."""

    def setup(self):
        gca_a = np.stack([_W0, _W1])
        gca_b = np.stack([_V0, _V1])
        _accux_gca(_W0, _W1, _V0, _V1)
        _try_gca_gca_intersection(_W0, _W1, _V0, _V1)
        gca_gca_intersection(gca_a, gca_b)
        self.gca_a = gca_a
        self.gca_b = gca_b

    def time_accux_gca_kernel(self):
        """Layer 1: pure numerical kernel."""
        _accux_gca(_W0, _W1, _V0, _V1)

    def time_try_gca_gca_intersection(self):
        """Layer 2: batch/status layer."""
        _try_gca_gca_intersection(_W0, _W1, _V0, _V1)

    def time_gca_gca_intersection(self):
        """Layer 3: dispatcher (full public API)."""
        gca_gca_intersection(self.gca_a, self.gca_b)


class GCAConstLatIntersection:
    """Benchmark all three layers of the GCA / constant-latitude intersection stack."""

    def setup(self):
        gca_cart = np.stack([_X1, _X2])
        _accux_constlat(_X1, _X2, _CONST_Z)
        _try_gca_const_lat_intersection(gca_cart, _CONST_Z)
        gca_const_lat_intersection(gca_cart, _CONST_Z)
        self.gca_cart = gca_cart

    def time_accux_constlat_kernel(self):
        """Layer 1: pure numerical kernel."""
        _accux_constlat(_X1, _X2, _CONST_Z)

    def time_try_gca_const_lat_intersection(self):
        """Layer 2: batch/status layer."""
        _try_gca_const_lat_intersection(self.gca_cart, _CONST_Z)

    def time_gca_const_lat_intersection(self):
        """Layer 3: dispatcher (full public API)."""
        gca_const_lat_intersection(self.gca_cart, _CONST_Z)


class PointInPolygonSphere:
    """Benchmark the spherical point-in-polygon kernel."""

    def setup(self):
        # Warm Numba
        _point_in_polygon_sphere(_Q_INSIDE, _POLY)
        _point_in_polygon_sphere(_Q_OUTSIDE, _POLY)

    def time_point_inside(self):
        _point_in_polygon_sphere(_Q_INSIDE, _POLY)

    def time_point_outside(self):
        _point_in_polygon_sphere(_Q_OUTSIDE, _POLY)
