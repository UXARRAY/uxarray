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
        # # In this testcase, we will only have one intersection point
        # gcr_cart = [ux.normalize_in_place([0.1,0.1,1]), ux.normalize_in_place([0.1,0.1,-1])]
        # const_lat = 0
        # res = ux.get_intersection_pt(gcr_cart, const_lat)
        # res = res[0] if res[0] != [-1, -1, -1] else res[1]
        # self.assertAlmostEqual(res[0] ** 2 + res[1] ** 2 + res[2] ** 2, 1,12)
        # self.assertAlmostEqual(np.dot(np.cross(gcr_cart[0], gcr_cart[1]),res),0,17)
        #
        # # A more complicate testcase that will have two intersection points
        # v0_rad = np.deg2rad(np.array([10, 40]))
        # v1_rad = np.deg2rad(np.array([150, 40]))
        # n1 = ux.convert_node_lonlat_rad_to_xyz(list(v0_rad))
        # n2 = ux.convert_node_lonlat_rad_to_xyz(list(v1_rad))
        # const_lat = 1
        # res = ux.get_intersection_pt([n1, n2], const_lat)
        # first_pt = res[0]
        # normal_vect = np.cross(n1, n2)
        #
        # self.assertAlmostEqual(first_pt[0] ** 2 + first_pt[1] ** 2 + first_pt[2] ** 2, 1,12)
        # self.assertAlmostEqual(np.dot(np.cross(n1, n2),first_pt),0,12)
        #
        # second_pt = res[1]
        # self.assertAlmostEqual(second_pt[0] ** 2 + second_pt[1] ** 2 + second_pt[2] ** 2, 1,12)
        # self.assertAlmostEqual(np.dot(np.cross(n1, n2), second_pt),0,12)
        #
        # The testcase that caused trouble before:
        n1 = [0.323619995813612, -0.4295599711942348, 0.843058912210295]
        n2 = [0.36874615152693996, -0.42199785145447993, 0.8282174165651636]
        res = ux.get_intersection_pt([n1, n2], 1.0)

        second_pt = res[1]
        self.assertAlmostEqual(second_pt[0] ** 2 + second_pt[1] ** 2 + second_pt[2] ** 2, 1,12)
        self.assertAlmostEqual(np.dot(np.cross(n1, n2), second_pt),0,12)
        face_lon_bound_max_rad = 5.497787143782138
        face_lon_bound_min_rad = 5.358046967298215
        second_pt_lonlat_rad = ux.convert_node_xyz_to_lonlat_rad(second_pt)
        self.assertLessEqual(second_pt_lonlat_rad[0],face_lon_bound_max_rad)
        self.assertGreaterEqual(second_pt_lonlat_rad[0], face_lon_bound_min_rad)



