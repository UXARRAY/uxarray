import os
from unittest import TestCase
import numpy as np
import uxarray as ux
from pathlib import Path
from uxarray.grid.coordinates import _populate_centroid_coord

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


class TestCentroids(TestCase):

    def test_centroids_from_mean_verts(self):
        # Create a triangle
        test_triangle = np.array([(0, 0, 1), (0, 0, -1), (1, 0, 0)])

        # Calculate the expected centroid
        expected_centroid = np.mean(test_triangle, axis=0)

        # Open the dataset and find the centroids
        grid = ux.open_grid(test_triangle)
        _populate_centroid_coord(grid)

        # Test the values of the calculate centroids
        self.assertEqual(expected_centroid[0], grid.Mesh2_face_cart_x)
        self.assertEqual(expected_centroid[1], grid.Mesh2_face_cart_y)
        self.assertEqual(expected_centroid[2], grid.Mesh2_face_cart_z)
