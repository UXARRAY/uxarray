"""
Purpose: tests related to angle calculations on a grid
"""

import numpy as np
import xarray as xr

import uxarray as ux


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
