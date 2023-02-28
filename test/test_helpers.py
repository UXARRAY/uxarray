import os
import numpy as np
import numpy.testing as nt
import random
import random
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux

try:
    import constants
except ImportError:
    from . import constants

# Data files
current_path = Path(os.path.dirname(os.path.realpath(__file__)))

exodus = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
ne8 = current_path / 'meshfiles' / "scrip" / "outCSne8" / 'outCSne8.nc'
err_tolerance = 1.0e-12


class TestIntegrate(TestCase):

    def test_face_area_coords(self):
        """Test function for helper function get_all_face_area_from_coords."""
        # Note: currently only testing one face, but this can be used to get area of multiple faces
        x = np.array([0.57735027, 0.57735027, -0.57735027])
        y = np.array([-5.77350269e-01, 5.77350269e-01, 5.77350269e-01])
        z = np.array([-0.57735027, -0.57735027, -0.57735027])
        face_nodes = np.array([[0, 1, 2]]).astype(constants.int_dtype)
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

        nodes_lonlat = [[5.430544880865939, 0.9759192936287691],
                        [5.497787143782138, 1.008861290690584],
                        [5.423907634712113, 1.0393506170556863],
                        [5.358046967298215, 1.0029457222814133],
                        [5.430544880865939, 0.9759192936287691]]
        intersections_pts_list_lonlat = []
        pt_lon_min = 3 * np.pi
        pt_lon_max = -3 * np.pi
        for i in range (0,4):
            n1_lonlat = nodes_lonlat[i]
            n2_lonlat = nodes_lonlat[i+1]
            n1 = ux.convert_node_lonlat_rad_to_xyz(nodes_lonlat[i])
            n2 = ux.convert_node_lonlat_rad_to_xyz(nodes_lonlat[i+1])
            intersections = ux.get_intersection_pt([n1, n2], 1)
            if intersections[0] == [-1, -1, -1] and intersections[1] == [-1, -1, -1]:
                # The constant latitude didn't cross this edge
                continue
            elif intersections[0] != [-1, -1, -1] and intersections[1] != [-1, -1, -1]:
                # The constant latitude goes across this edge ( 1 in and 1 out):
                pts1_lonlat = ux.convert_node_xyz_to_lonlat_rad(intersections[0])
                pts2_lonlat = ux.convert_node_xyz_to_lonlat_rad(intersections[1])
                intersections_pts_list_lonlat.append(ux.convert_node_xyz_to_lonlat_rad(intersections[0]))
                intersections_pts_list_lonlat.append(ux.convert_node_xyz_to_lonlat_rad(intersections[1]))
            else:
                if intersections[0] != [-1, -1, -1]:
                    intersections_pts_list_lonlat.append(ux.convert_node_xyz_to_lonlat_rad(intersections[0]))
                else:
                    intersections_pts_list_lonlat.append(ux.convert_node_xyz_to_lonlat_rad(intersections[1]))
        if len(intersections_pts_list_lonlat) == 2:
            [pt_lon_min, pt_lon_max] = np.sort([intersections_pts_list_lonlat[0][0], intersections_pts_list_lonlat[1][0]])
        else:
            pass
        cur_face_mag_rad = pt_lon_max - pt_lon_min
        self.assertLessEqual(cur_face_mag_rad, np.pi)






class TestGridCenter(TestCase):

    def test_grid_center(self):
        """Calculates if the calculated center point of a grid box is the same
        as a given value for the same dataset."""
        ds_ne8 = xr.open_dataset(ne8)

        # select actual center_lat/lon
        scrip_center_lon = ds_ne8['grid_center_lon']
        scrip_center_lat = ds_ne8['grid_center_lat']

        # Calculate the center_lat/lon using same dataset's corner_lat/lon
        calc_center = ux.grid_center_lat_lon(ds_ne8)
        calc_lat = calc_center[0]
        calc_lon = calc_center[1]

        # Test that calculated center_lat/lon is the same as actual center_lat/lon
        np.testing.assert_array_almost_equal(scrip_center_lat, calc_lat)
        np.testing.assert_array_almost_equal(scrip_center_lon, calc_lon)


class TestCoordinatesConversion(TestCase):

    def test_normalize_in_place(self):
        [x, y, z] = ux.helpers._normalize_in_place(
            [random.random(), random.random(),
             random.random()])
        self.assertLessEqual(np.absolute(np.sqrt(x * x + y * y + z * z) - 1),
                             err_tolerance)

    def test_convert_node_xyz_to_lonlat_rad(self):
        [x, y, z] = ux.helpers._normalize_in_place([
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ])
        [lon, lat] = ux.helpers._convert_node_xyz_to_lonlat_rad([x, y, z])
        [new_x, new_y,
         new_z] = ux.helpers._convert_node_lonlat_rad_to_xyz([lon, lat])
        self.assertLessEqual(np.absolute(new_x - x), err_tolerance)
        self.assertLessEqual(np.absolute(new_y - y), err_tolerance)
        self.assertLessEqual(np.absolute(new_z - z), err_tolerance)

    def test_convert_node_latlon_rad_to_xyz(self):
        [lon, lat] = [
            random.uniform(0, 2 * np.pi),
            random.uniform(-0.5 * np.pi, 0.5 * np.pi)
        ]
        [x, y, z] = ux.helpers._convert_node_lonlat_rad_to_xyz([lon, lat])
        [new_lon,
         new_lat] = ux.helpers._convert_node_xyz_to_lonlat_rad([x, y, z])
        self.assertLessEqual(np.absolute(new_lon - lon), err_tolerance)
        self.assertLessEqual(np.absolute(new_lat - lat), err_tolerance)
