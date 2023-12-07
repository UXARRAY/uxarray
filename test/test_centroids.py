import os
from unittest import TestCase
import numpy as np
import numpy.testing as nt
import uxarray as ux
from pathlib import Path
from uxarray.grid.coordinates import _populate_centroid_coord, normalize_in_place

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_CSne8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"


class TestCentroids(TestCase):

    def test_centroids_from_mean_verts_triangle(self):
        """Test finding the centroid of a triangle."""
        # Create a triangle
        test_triangle = np.array([(0, 0, 1), (0, 0, -1), (1, 0, 0)])

        # Calculate the expected centroid
        expected_centroid = np.mean(test_triangle, axis=0)
        [norm_x, norm_y, norm_z] = normalize_in_place(
            [expected_centroid[0], expected_centroid[1], expected_centroid[2]])

        # Open the dataset and find the centroids
        grid = ux.open_grid(test_triangle)
        _populate_centroid_coord(grid)

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
        [norm_x, norm_y, norm_z] = normalize_in_place(
            [expected_centroid[0], expected_centroid[1], expected_centroid[2]])

        # Open the dataset and find the centroids
        grid = ux.open_grid(test_triangle)
        _populate_centroid_coord(grid)

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

        _populate_centroid_coord(uxgrid, repopulate=True)

        computed_face_x = (uxgrid.face_lon.values + 180) % 360 - 180
        computed_face_y = uxgrid.face_lat.values

        nt.assert_array_almost_equal(expected_face_x, computed_face_x)
        nt.assert_array_almost_equal(expected_face_y, computed_face_y)
