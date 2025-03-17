import os
import numpy as np
import numpy.testing as nt
import uxarray as ux
from pathlib import Path
from uxarray.grid.coordinates import (
    _populate_face_centroids,
    _populate_edge_centroids,
    _populate_face_centerpoints,
    _is_inside_circle,
    _circle_from_three_points,
    _circle_from_two_points,
    _normalize_xyz,
)

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_CSne8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
mpasfile_QU = current_path / "meshfiles" / "mpas" / "QU" / "mesh.QU.1920km.151026.nc"

def test_centroids_from_mean_verts_triangle():
    """Test finding the centroid of a triangle."""
    test_triangle = np.array([(0, 0, 1), (0, 0, -1), (1, 0, 0)])
    expected_centroid = np.mean(test_triangle, axis=0)
    norm_x, norm_y, norm_z = _normalize_xyz(
        expected_centroid[0], expected_centroid[1], expected_centroid[2]
    )

    grid = ux.open_grid(test_triangle, latlon=False)
    _populate_face_centroids(grid)

    assert norm_x == grid.face_x
    assert norm_y == grid.face_y
    assert norm_z == grid.face_z

def test_centroids_from_mean_verts_pentagon():
    """Test finding the centroid of a pentagon."""
    test_polygon = np.array([(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0), (125, 125, 1)])
    expected_centroid = np.mean(test_polygon, axis=0)
    norm_x, norm_y, norm_z = _normalize_xyz(
        expected_centroid[0], expected_centroid[1], expected_centroid[2]
    )

    grid = ux.open_grid(test_polygon, latlon=False)
    _populate_face_centroids(grid)

    assert norm_x == grid.face_x
    assert norm_y == grid.face_y
    assert norm_z == grid.face_z

def test_centroids_from_mean_verts_scrip():
    """Test computed centroid values compared to values from a SCRIP dataset."""
    uxgrid = ux.open_grid(gridfile_CSne8)

    expected_face_x = uxgrid.face_lon.values
    expected_face_y = uxgrid.face_lat.values

    uxgrid.construct_face_centers(method="cartesian average")

    computed_face_x = uxgrid.face_lon.values
    computed_face_y = uxgrid.face_lat.values

    nt.assert_array_almost_equal(expected_face_x, computed_face_x)
    nt.assert_array_almost_equal(expected_face_y, computed_face_y)

def test_edge_centroids_from_triangle():
    """Test finding the centroid of a triangle."""
    test_triangle = np.array([(0, 0, 0), (-1, 1, 0), (-1, -1, 0)])
    grid = ux.open_grid(test_triangle, latlon=False)
    _populate_edge_centroids(grid)

    centroid_x = np.mean(grid.node_x[grid.edge_node_connectivity[0][0:]])
    centroid_y = np.mean(grid.node_y[grid.edge_node_connectivity[0][0:]])
    centroid_z = np.mean(grid.node_z[grid.edge_node_connectivity[0][0:]])

    assert centroid_x == grid.edge_x[0]
    assert centroid_y == grid.edge_y[0]
    assert centroid_z == grid.edge_z[0]

def test_edge_centroids_from_mpas():
    """Test computed centroid values compared to values from a MPAS dataset."""
    uxgrid = ux.open_grid(mpasfile_QU)

    expected_edge_lon = uxgrid.edge_lon.values
    expected_edge_lat = uxgrid.edge_lat.values

    _populate_edge_centroids(uxgrid, repopulate=True)

    computed_edge_lon = (uxgrid.edge_lon.values + 180) % 360 - 180
    computed_edge_lat = uxgrid.edge_lat.values

    nt.assert_array_almost_equal(expected_edge_lon, computed_edge_lon)
    nt.assert_array_almost_equal(expected_edge_lat, computed_edge_lat)

def test_circle_from_two_points():
    """Test creation of circle from 2 points."""
    p1 = (0, 0)
    p2 = (0, 90)
    center, radius = _circle_from_two_points(p1, p2)

    expected_center = (0.0, 45.0)
    expected_radius = np.deg2rad(45.0)

    assert np.allclose(center, expected_center), f"Expected center {expected_center}, but got {center}"
    assert np.allclose(radius, expected_radius), f"Expected radius {expected_radius}, but got {radius}"

def test_circle_from_three_points():
    """Test creation of circle from 3 points."""
    p1 = (0, 0)
    p2 = (0, 90)
    p3 = (90, 0)
    center, radius = _circle_from_three_points(p1, p2, p3)
    expected_radius = np.deg2rad(45.0)
    expected_center = (30.0, 30.0)

    assert np.allclose(center, expected_center), f"Expected center {expected_center}, but got {center}"
    assert np.allclose(radius, expected_radius), f"Expected radius {expected_radius}, but got {radius}"

def test_is_inside_circle():
    """Test if a points is inside the circle."""
    circle = ((0.0, 0.0), 1)  # Center at lon/lat with a radius in radians

    point_inside = (30.0, 30.0)
    point_outside = (90.0, 0.0)

    assert _is_inside_circle(circle, point_inside), f"Point {point_inside} should be inside the circle."
    assert not _is_inside_circle(circle, point_outside), f"Point {point_outside} should be outside the circle."

def test_face_centerpoint():
    """Use points from an actual spherical face and get the centerpoint."""
    points = np.array([
        (-35.26438968, -45.0),
        (-36.61769496, -42.0),
        (-33.78769181, -42.0),
        (-32.48416571, -45.0)
    ])
    uxgrid = ux.open_grid(points, latlon=True)

    ctr_lon = uxgrid.face_lon.values[0]
    ctr_lat = uxgrid.face_lat.values[0]

    uxgrid.construct_face_centers(method="welzl")

    nt.assert_array_almost_equal(ctr_lon, uxgrid.face_lon.values[0], decimal=2)
    nt.assert_array_almost_equal(ctr_lat, uxgrid.face_lat.values[0], decimal=2)
