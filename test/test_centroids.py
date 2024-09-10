import os
from unittest import TestCase
import numpy as np
import numpy.testing as nt
import uxarray as ux
from pathlib import Path
from uxarray.grid.coordinates import _populate_face_centroids, _populate_edge_centroids, _populate_face_centerpoints, _is_inside_circle, _circle_from_three_points, _circle_from_two_points
from uxarray.grid.utils import _normalize_xyz

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_CSne8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
mpasfile_QU = current_path / "meshfiles" / "mpas" / "QU" / "mesh.QU.1920km.151026.nc"


class TestCentroids(TestCase):

    def test_centroids_from_mean_verts_triangle(self):
        """Test finding the centroid of a triangle."""
        # Create a triangle
        test_triangle = np.array([(0, 0, 1), (0, 0, -1), (1, 0, 0)])

        # Calculate the expected centroid
        expected_centroid = np.mean(test_triangle, axis=0)
        norm_x, norm_y, norm_z = _normalize_xyz(
            expected_centroid[0], expected_centroid[1], expected_centroid[2])

        # Open the dataset and find the centroids
        grid = ux.open_grid(test_triangle)
        _populate_face_centroids(grid)

        # Test the values of the calculate centroids
        self.assertEqual(norm_x, grid.face_x)
        self.assertEqual(norm_y, grid.face_y)
        self.assertEqual(norm_z, grid.face_z)

    def test_centroids_from_mean_verts_pentagon(self):
        """Test finding the centroid of a pentagon."""

        # Create a polygon
        test_triangle = np.array([(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, 1, 0),
                                  (125, 125, 1)])

        # Calculate the expected centroid
        expected_centroid = np.mean(test_triangle, axis=0)
        norm_x, norm_y, norm_z = _normalize_xyz(
            expected_centroid[0], expected_centroid[1], expected_centroid[2])

        # Open the dataset and find the centroids
        grid = ux.open_grid(test_triangle)
        _populate_face_centroids(grid)

        # Test the values of the calculate centroids
        self.assertEqual(norm_x, grid.face_x)
        self.assertEqual(norm_y, grid.face_y)
        self.assertEqual(norm_z, grid.face_z)

    def test_centroids_from_mean_verts_scrip(self):
        """Test computed centroid values compared to values from a SCRIP
        dataset."""

        uxgrid = ux.open_grid(gridfile_CSne8)

        expected_face_x = uxgrid.face_lon.values
        expected_face_y = uxgrid.face_lat.values

        # _populate_face_centroids(uxgrid, repopulate=True)
        uxgrid.construct_face_centers(method="cartesian average")

        # computed_face_x = (uxgrid.face_lon.values + 180) % 360 - 180
        computed_face_x = uxgrid.face_lon.values
        computed_face_y = uxgrid.face_lat.values

        nt.assert_array_almost_equal(expected_face_x, computed_face_x)
        nt.assert_array_almost_equal(expected_face_y, computed_face_y)

    def test_edge_centroids_from_triangle(self):
        """Test finding the centroid of a triangle."""
        # Create a triangle
        test_triangle = np.array([(0, 0, 0), (-1, 1, 0), (-1, -1, 0)])

        # Open the dataset and find the centroids
        grid = ux.open_grid(test_triangle)
        _populate_edge_centroids(grid)

        # compute edge_xyz for first edge
        centroid_x = np.mean(grid.node_x[grid.edge_node_connectivity[0][0:]])
        centroid_y = np.mean(grid.node_y[grid.edge_node_connectivity[0][0:]])
        centroid_z = np.mean(grid.node_z[grid.edge_node_connectivity[0][0:]])

        # Test the values of computed first edge centroid and the populated one
        self.assertEqual(centroid_x, grid.edge_x[0])
        self.assertEqual(centroid_y, grid.edge_y[0])
        self.assertEqual(centroid_z, grid.edge_z[0])

    def test_edge_centroids_from_mpas(self):
        """Test computed centroid values compared to values from a MPAS
        dataset."""

        uxgrid = ux.open_grid(mpasfile_QU)

        expected_edge_lon = uxgrid.edge_lon.values
        expected_edge_lat = uxgrid.edge_lat.values

        _populate_edge_centroids(uxgrid, repopulate=True)

        computed_edge_lon = (uxgrid.edge_lon.values + 180) % 360 - 180
        computed_edge_lat = uxgrid.edge_lat.values

        nt.assert_array_almost_equal(expected_edge_lon, computed_edge_lon)
        nt.assert_array_almost_equal(expected_edge_lat, computed_edge_lat)

class TestCenterPoints(TestCase):

    def test_circle_from_two_points(self):
        """Test creation of circle from 2 points."""
        p1 = (0, 0)
        p2 = (0, 90)
        center, radius = _circle_from_two_points(p1, p2)

        # The expected radius in radians should be half the angle between the two vectors
        expected_center = (0.0, 45.0)
        expected_radius = np.deg2rad(45.0)

        assert np.allclose(center, expected_center), f"Expected center {expected_center}, but got {center}"
        assert np.allclose(radius, expected_radius), f"Expected radius {expected_radius}, but got {radius}"

    def test_circle_from_three_points(self):
        """Test creation of circle from 3 points."""
        p1 = (0, 0)
        p2 = (0, 90)
        p3 = (90, 0)
        center, radius = _circle_from_three_points(p1, p2, p3)
        expected_radius = np.deg2rad(45.0)
        expected_center = (30.0, 30.0)

        assert np.allclose(center, expected_center), f"Expected center {expected_center}, but got {center}"
        assert np.allclose(radius, expected_radius), f"Expected radius {expected_radius}, but got {radius}"

    def test_is_inside_circle(self):
        """Test if a points is inside the circle."""
        # Define the circle
        circle = ((0.0, 0.0), 1)  # Center at lon/lat with a radius in radians (angular measure of the radius)

        # Define test points
        point_inside = (30.0, 30.0)  # Should be inside the circle
        point_outside = (90.0, 0.0)  # Should be outside the circle

        # Test _is_inside_circle function
        assert _is_inside_circle(circle, point_inside), f"Point {point_inside} should be inside the circle."
        assert not _is_inside_circle(circle, point_outside), f"Point {point_outside} should be outside the circle."

    def test_face_centerpoint(self):
        """Use points from an actual spherical face and get the centerpoint."""

        points = np.array([(-35.26438968, -45.0), (-36.61769496, -42.0), (-33.78769181, -42.0), (-32.48416571, -45.0)])
        uxgrid = ux.open_grid(points, latlon=True)

        # Uses the @property from get face_lon/lat - default is average or centroid
        ctr_lon = uxgrid.face_lon.values[0]
        ctr_lat = uxgrid.face_lat.values[0]

        # now explicitly get the centerpoints stored to face_lon/lat using welzl's centerpoint algorithm
        uxgrid.construct_face_centers(method = "welzl")

        # Test the values of the calculated centerpoint, giving high tolerance of two decimal place
        nt.assert_array_almost_equal(ctr_lon, uxgrid.face_lon.values[0], decimal=2)
        nt.assert_array_almost_equal(ctr_lat, uxgrid.face_lat.values[0], decimal=2)
