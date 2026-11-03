"""
Purpose: tests related to angle calculations on a grid
"""

import numpy as np
import pytest
import xarray as xr

import uxarray as ux
from uxarray.grid.arcs import _angle_of_2_vectors
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.utils import _small_angle_of_2_vectors


def test_face_node_angles_triangle():
    """ensure Grid.compute_face_node_angles() works as expected for a simple ~30,60,90 triangle."""
    # make a tiny triangle with known angles (90,60,30 degrees):
    # (n1)
    #  |   %%
    #  |       %%
    # (n0) ------ (n2)
    node_lon = [0, 0, np.sqrt(3)]
    node_lat = [0, 1, 0]
    face_node_connectivity = [[0, 1, 2]]
    grid = ux.Grid.from_topology(node_lon, node_lat, face_node_connectivity)
    angles_rad = grid.compute_face_node_angles()
    angles_deg = grid.compute_face_node_angles(degrees=True)
    assert np.allclose(np.rad2deg(angles_rad), angles_deg)
    angles_uxarr = grid.compute_face_node_angles(as_uxarray=True)
    assert isinstance(angles_rad, xr.DataArray)
    assert isinstance(angles_uxarr, ux.UxDataArray)
    assert np.all(angles_rad == angles_uxarr)
    angle_at_n0 = angles_deg.isel(n_face=0, n_max_face_nodes=0)
    angle_at_n1 = angles_deg.isel(n_face=0, n_max_face_nodes=1)
    angle_at_n2 = angles_deg.isel(n_face=0, n_max_face_nodes=2)
    assert np.isclose(angle_at_n0, 90.0, atol=0, rtol=1e-16)  # rad2deg(arctan2(any_value, 0)) == 90
    assert np.isclose(angle_at_n1, 60.0, atol=1e-2, rtol=0)   # basically 60 degrees
    assert np.isclose(angle_at_n2, 30.0, atol=1e-2, rtol=0)   # basically 30 degrees
    # on a unit sphere, spherical excess == face area, via Girard's theorem.
    spherical_excess = angles_rad.sum('n_max_face_nodes') - np.pi
    face_areas = grid.compute_face_areas()
    assert np.allclose(spherical_excess, face_areas, atol=0, rtol=1e-12)

def test_face_node_angles_hexagons_and_pentagons():
    """ensure face_node_angles works as expected on grids with hexagons and pentagons"""
    grid = ux.tutorial.open_grid('quad-hexagon')  # has multiple faces, all hexagons.
    angles_deg = grid.compute_face_node_angles(degrees=True)
    # every hexagon in this grid is close to regular (all 120 degree angles):
    regular_hex_deviation = angles_deg - 120.0
    assert np.max(np.abs(regular_hex_deviation)) < 4.0
    # generalized spherical excess formula uses (n - 2) * np.pi; n==6 for all of these faces
    angles = grid.compute_face_node_angles()  # (need to use radians for this formula)
    spherical_excess = angles.sum('n_max_face_nodes') - (6 - 2) * np.pi
    face_areas = grid.compute_face_areas()
    assert np.allclose(spherical_excess, face_areas, atol=0, rtol=1e-10)

    # now test a grid which has pentagons too,
    # to ensure the implementation works even when the number of nodes per face varies.
    grid = ux.tutorial.open_grid('mpas-QU-480')
    # not all close to regular so don't try to check that.
    # ensure nan values wherever n_max_face_nodes dimension is larger than n_nodes_per_face
    angles = grid.compute_face_node_angles()
    assert not np.all(grid.n_nodes_per_face == grid.n_max_face_nodes)
    should_have_nans = angles.where(grid.n_nodes_per_face < grid.n_max_face_nodes, drop=True)
    assert should_have_nans.size > 0
    should_be_nans = should_have_nans.isel(n_max_face_nodes = -1)
    assert np.all(np.isnan(should_be_nans))
    # generalized spherical excess formula uses (n_nodes_per_face - 2) * np.pi
    spherical_excess = angles.sum('n_max_face_nodes') - (grid.n_nodes_per_face - 2) * np.pi
    face_areas = grid.compute_face_areas()
    assert np.allclose(spherical_excess, face_areas, atol=0, rtol=1e-9)

def test_equiangle_skewness():
    """just some tests about Grid.compute_skewness(method="equiangle")..."""

    ## spot check with some triangles
    # make a tiny 90,60,30 degrees triangle:
    # (n1)
    #  |   %%
    #  |       %%
    # (n0) ------ (n2)
    node_lon = [0, 0, np.sqrt(3)]
    node_lat = [0, 1, 0]
    face_node_connectivity = [[0, 1, 2]]
    grid = ux.Grid.from_topology(node_lon, node_lat, face_node_connectivity)
    # expect skewness = max((Amax - Areg) / (180 degrees - Areg), (Areg - Amin) / Areg)
    # here, Areg ~= 60 degrees (it's a tiny triangle), Amax = 90 degrees, Amin = 30 degrees
    #   --> max((90 - 60) / (180 - 60), (60 - 30) / 60) --> max(0.25, 0.5)
    skewness = grid.compute_skewness(method="equiangle")
    assert np.isclose(skewness, 0.5, atol=1e-4, rtol=0)

    ## much larger "would-be 90,60,30" triangle, to ensure accounting for spherical geometry.
    # It's actually 90, 68.6, 36.2 degrees, due to spherical geometry.
    node_lon = [0, 0, 30*np.sqrt(3)]
    node_lat = [0, 30, 0]
    face_node_connectivity = [[0, 1, 2]]
    grid = ux.Grid.from_topology(node_lon, node_lat, face_node_connectivity)
    skewness = grid.compute_skewness(method="equiangle")
    assert not np.isclose(skewness, 0.5, atol=1e-2, rtol=0)
    assert 0.42 < skewness < 0.46  # hard-coding "expected" answer.
    # If using correct angles but not spherical geometry in Areg, would assume Areg = 60
    #   --> skewness = max((90 - 68.6) / (180 - 68.6), (68.6 - 36.2) / 68.6) --> max(0.192, 0.472)
    # Therefore, the test above is indeed sensitive to
    #   "does the skewness formula actually use the proper Areg based on spherical geometry?"

    ## spot check of grid with "not all faces have same number of nodes"
    grid = ux.tutorial.open_grid('mpas-QU-480')
    skewness = grid.compute_skewness(method="equiangle")
    assert isinstance(skewness, xr.DataArray)
    # there happens to be at least one very non-skew shape in this grid
    assert 0 < skewness.min() < 1e-8
    # there aren't any very skew shapes in this grid
    assert skewness.max() < 0.15
    # skewness is well defined for all faces
    assert not np.any(skewness.isnull())


def test_angle_of_2_vectors():
    # Test the angle between two vectors
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert pytest.approx(_angle_of_2_vectors(v1, v2)) == np.pi / 2.0

    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    assert pytest.approx(_angle_of_2_vectors(v1, v2)) == 0.0

def test_angle_of_2_vectors_180_degree():
    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(0.0),
                                np.deg2rad(0.0)),
        _lonlat_rad_to_xyz(np.deg2rad(181.0),
                                np.deg2rad(0.0))
    ])

    res = _angle_of_2_vectors(GCR1_cart[0], GCR1_cart[1])

    # The angle between the two vectors should be 181 degree
    assert pytest.approx(res, abs=1e-8) == np.deg2rad(181.0)

    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                np.deg2rad(89.0)),
        _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                np.deg2rad(-10.0))
    ])

    res = _angle_of_2_vectors(GCR1_cart[0], GCR1_cart[1])

    # The angle between the two vectors should be 99 degrees
    assert pytest.approx(res, abs=1e-8) == np.deg2rad(89.0+10.0)


def _exact_small_angle(u, v):
    """Angle between two float vectors, for angles well below 1 rad.

    The cross and dot products are formed exactly in rational arithmetic, then
    atan(|u x v| / (u . v)) is evaluated with a 50-digit Taylor series.
    """
    from decimal import Decimal, localcontext
    from fractions import Fraction

    u = [Fraction(float(c)) for c in u]
    v = [Fraction(float(c)) for c in v]
    cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    cross2 = sum(c * c for c in cross)
    dot = sum(a * b for a, b in zip(u, v))
    with localcontext() as ctx:
        ctx.prec = 50
        t = (Decimal(cross2.numerator) / Decimal(cross2.denominator)).sqrt() / (
            Decimal(dot.numerator) / Decimal(dot.denominator))
        term, total, k = t, Decimal(0), 0
        while abs(term) > Decimal(10) ** -45:
            total += term / (2 * k + 1) * (-1) ** k
            term *= t * t
            k += 1
        return float(total)


def test_small_angle_of_2_vectors_special_values():
    assert _small_angle_of_2_vectors((1.0, 0.0, 0.0), (0.0, 3.0, 0.0)) == pytest.approx(np.pi / 2, rel=1e-15)
    assert _small_angle_of_2_vectors((2.0, 0.0, 0.0), (0.5, 0.0, 0.0)) == 0.0
    assert _small_angle_of_2_vectors((0.0, 0.0, 1.0), (0.0, 0.0, -7.0)) == pytest.approx(np.pi, rel=1e-15)
    assert _small_angle_of_2_vectors((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)) == 0.0


@pytest.mark.parametrize("theta, rtol", [(1e-3, 1e-14), (1e-5, 1e-14), (1e-7, 1e-14), (1e-9, 1e-13)])
def test_small_angle_of_2_vectors_accuracy(theta, rtol):
    """Nearly parallel vectors of different, arbitrary lengths.

    A naive evaluation of v|u| - u|v| loses accuracy as eps / theta (about 1e-7
    at theta = 1e-9); with compensated products it stays within a few ulps,
    plus a second-order (eps / theta)^2 term from the rounded norms.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        u = rng.normal(size=3)
        w = np.cross(u, rng.normal(size=3))
        v = u / np.linalg.norm(u) * np.cos(theta) + w / np.linalg.norm(w) * np.sin(theta)
        u, v = u * rng.uniform(0.1, 10.0), v * rng.uniform(0.1, 10.0)
        expected = _exact_small_angle(u, v)
        assert _small_angle_of_2_vectors(u, v) == pytest.approx(expected, rel=rtol, abs=0.0)
        assert _small_angle_of_2_vectors(v, u) == pytest.approx(expected, rel=rtol, abs=0.0)
