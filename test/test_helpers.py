import os
import numpy as np
import numpy.testing as nt
import random

from unittest import TestCase
from pathlib import Path

import uxarray as ux

try:
    import constants
except ImportError:
    from . import constants

# Data files
current_path = Path(os.path.dirname(os.path.realpath(__file__)))

exodus = current_path / "meshfiles" / "outCSne8.g"


class TestIntegrate(TestCase):

    def test_face_area_coords(self):
        """Test function for helper function get_all_face_area_from_coords."""
        # Note: currently only testing one face, but this can be used to get area of multiple faces
        x = np.array([0.57735027, 0.57735027, -0.57735027])
        y = np.array([-5.77350269e-01, 5.77350269e-01, 5.77350269e-01])
        z = np.array([-0.57735027, -0.57735027, -0.57735027])
        face_nodes = np.array([[0, 1, 2]])
        area = ux.get_all_face_area_from_coords(x,
                                                y,
                                                z,
                                                face_nodes,
                                                3,
                                                coords_type="cartesian")
        nt.assert_almost_equal(area, constants.TRI_AREA, decimal=1)

    def test_calculate_face_area(self):
        """Test function for helper function calculate_face_area - only one face."""
        # Note: currently only testing one face, but this can be used to get area of multiple faces
        # Also note, this does not need face_nodes, assumes nodes are in counterclockwise orientation
        x = np.array([0.57735027, 0.57735027, -0.57735027])
        y = np.array([-5.77350269e-01, 5.77350269e-01, 5.77350269e-01])
        z = np.array([-0.57735027, -0.57735027, -0.57735027])
        area = ux.calculate_face_area(x, y, z, "gaussian", 5, "cartesian")
        nt.assert_almost_equal(area, constants.TRI_AREA, decimal=3)

    def test_quadrature(self):
        order = 1
        dG, dW = ux.get_tri_quadratureDG(order)
        G = np.array([[0.33333333, 0.33333333, 0.33333333]])
        W = np.array([1.0])

        np.testing.assert_array_almost_equal(G, dG)
        np.testing.assert_array_almost_equal(W, dW)

        dG, dW = ux.get_gauss_quadratureDG(order)
        G = np.array([[0.5]])
        W = np.array([1.0])

        np.testing.assert_array_almost_equal(G, dG)
        np.testing.assert_array_almost_equal(W, dW)

    def test_normalize_in_place(self):
        for i in range(0, 10):
            [x, y, z] = ux.normalize_in_place(
                [random.random(),
                 random.random(),
                 random.random()])
            self.assertLessEqual(
                np.absolute(np.sqrt(x * x + y * y + z * z) - 1), 1.0e-12)

    def test_convert_node_xyz_to_lonlat_rad(self):
        for i in range(0, 10):
            [x, y, z] = ux.normalize_in_place([
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(-1, 1)
            ])
            [lon, lat] = ux.convert_node_xyz_to_lonlat_rad([x, y, z])
            [new_x, new_y,
             new_z] = ux.convert_node_lonlat_rad_to_xyz([lon, lat])
            self.assertLessEqual(np.absolute(new_x - x), 1.0e-12)
            self.assertLessEqual(np.absolute(new_y - y), 1.0e-12)
            self.assertLessEqual(np.absolute(new_z - z), 1.0e-12)

    def test_convert_node_latlon_rad_to_xyz(self):
        for i in range(0, 10):
            [lon, lat] = [
                random.uniform(0, 2 * np.pi),
                random.uniform(-0.5 * np.pi, 0.5 * np.pi)
            ]
            [x, y, z] = ux.convert_node_lonlat_rad_to_xyz([lon, lat])
            [new_lon, new_lat] = ux.convert_node_xyz_to_lonlat_rad([x, y, z])
            self.assertLessEqual(np.absolute(new_lon - lon), 1.0e-12)
            self.assertLessEqual(np.absolute(new_lat - lat), 1.0e-12)


    def test_get_intersection_pt(self):
        gcr_cart = [ux.normalize_in_place([0.1,0.1,1]), ux.normalize_in_place([0.1,0.1,-1])]
        const_lat = 0
        res = ux.get_intersection_pt(gcr_cart, const_lat)
        self.assertEqual(res[0] ** 2 + res[1] ** 2 + const_lat ** 2, 1)
        self.assertAlmostEqual(np.dot(np.cross(gcr_cart[0], gcr_cart[1]),[res[0], res[1],const_lat]),1.0e-12)


