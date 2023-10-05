import os
import numpy as np
import numpy.testing as nt
import random
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux

from uxarray.grid.connectivity import _replace_fill_values
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

from uxarray.grid.coordinates import node_lonlat_rad_to_xyz
from uxarray.grid.lines import point_within_gca, _angle_of_2_vectors, in_between

try:
    import constants
except ImportError:
    from . import constants

# Data files
current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_exo_CSne8 = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
gridfile_scrip_CSne8 = current_path / 'meshfiles' / "scrip" / "outCSne8" / 'outCSne8.nc'

err_tolerance = 1.0e-12


class TestIntegrate(TestCase):

    def test_face_area_coords(self):
        """Test function for helper function get_all_face_area_from_coords."""
        # Note: currently only testing one face, but this can be used to get area of multiple faces
        x = np.array([0.57735027, 0.57735027, -0.57735027])
        y = np.array([-5.77350269e-01, 5.77350269e-01, 5.77350269e-01])
        z = np.array([-0.57735027, -0.57735027, -0.57735027])

        face_nodes = np.array([[0, 1, 2]]).astype(INT_DTYPE)
        face_dimension = np.array([3], dtype=INT_DTYPE)

        area = ux.grid.area.get_all_face_area_from_coords(
            x, y, z, face_nodes, face_dimension, 3, coords_type="cartesian")

        nt.assert_almost_equal(area, constants.TRI_AREA, decimal=1)

    def test_calculate_face_area(self):
        """Test function for helper function calculate_face_area - only one face."""
        # Note: currently only testing one face, but this can be used to get area of multiple faces
        # Also note, this does not need face_nodes, assumes nodes are in counterclockwise orientation
        x = np.array([0.57735027, 0.57735027, -0.57735027])
        y = np.array([-5.77350269e-01, 5.77350269e-01, 5.77350269e-01])
        z = np.array([-0.57735027, -0.57735027, -0.57735027])

        area = ux.grid.area.calculate_face_area(x, y, z, "gaussian", 5,
                                                "cartesian")

        nt.assert_almost_equal(area, constants.TRI_AREA, decimal=3)

    def test_quadrature(self):
        order = 1
        dG, dW = ux.grid.area.get_tri_quadratureDG(order)
        G = np.array([[0.33333333, 0.33333333, 0.33333333]])
        W = np.array([1.0])

        np.testing.assert_array_almost_equal(G, dG)
        np.testing.assert_array_almost_equal(W, dW)

        dG, dW = ux.grid.area.get_gauss_quadratureDG(order)

        G = np.array([[0.5]])
        W = np.array([1.0])

        np.testing.assert_array_almost_equal(G, dG)
        np.testing.assert_array_almost_equal(W, dW)


class TestGridCenter(TestCase):

    def test_grid_center(self):
        """Calculates if the calculated center point of a grid box is the same
        as a given value for the same dataset."""
        ds_scrip_CSne8 = xr.open_dataset(gridfile_scrip_CSne8)

        # select actual center_lat/lon
        scrip_center_lon = ds_scrip_CSne8['grid_center_lon']
        scrip_center_lat = ds_scrip_CSne8['grid_center_lat']

        # Calculate the center_lat/lon using same dataset's corner_lat/lon
        calc_center = ux.io._scrip.grid_center_lat_lon(ds_scrip_CSne8)
        calc_lat = calc_center[0]
        calc_lon = calc_center[1]

        # Test that calculated center_lat/lon is the same as actual center_lat/lon
        np.testing.assert_array_almost_equal(scrip_center_lat, calc_lat)
        np.testing.assert_array_almost_equal(scrip_center_lon, calc_lon)


class TestCoordinatesConversion(TestCase):

    def test_normalize_in_place(self):
        [x, y, z] = ux.grid.coordinates.normalize_in_place(
            [random.random(), random.random(),
             random.random()])

        self.assertLessEqual(np.absolute(np.sqrt(x * x + y * y + z * z) - 1),
                             err_tolerance)

    def test_node_xyz_to_lonlat_rad(self):
        [x, y, z] = ux.grid.coordinates.normalize_in_place([
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ])

        [lon, lat] = ux.grid.coordinates.node_xyz_to_lonlat_rad([x, y, z])
        [new_x, new_y,
         new_z] = ux.grid.coordinates.node_lonlat_rad_to_xyz([lon, lat])

        self.assertLessEqual(np.absolute(new_x - x), err_tolerance)
        self.assertLessEqual(np.absolute(new_y - y), err_tolerance)
        self.assertLessEqual(np.absolute(new_z - z), err_tolerance)

    def test_node_latlon_rad_to_xyz(self):
        [lon, lat] = [
            random.uniform(0, 2 * np.pi),
            random.uniform(-0.5 * np.pi, 0.5 * np.pi)
        ]

        [x, y, z] = ux.grid.coordinates.node_lonlat_rad_to_xyz([lon, lat])

        [new_lon,
         new_lat] = ux.grid.coordinates.node_xyz_to_lonlat_rad([x, y, z])

        self.assertLessEqual(np.absolute(new_lon - lon), err_tolerance)
        self.assertLessEqual(np.absolute(new_lat - lat), err_tolerance)


class TestConstants(TestCase):
    # DTYPE as set in constants.py
    expected_int_dtype = INT_DTYPE

    # INT_FILL_VALUE as set in constants.py
    fv = INT_FILL_VALUE

    def test_invalid_indexing(self):
        """Tests if the current INT_DTYPE and INT_FILL_VALUE throw the correct
        errors when indexing."""
        dummy_data = np.array([1, 2, 3, 4])

        invalid_indices = np.array([self.fv, self.fv], dtype=INT_DTYPE)
        invalid_index = self.fv

        # invalid index/indices should throw an Index Error
        with self.assertRaises(IndexError):
            dummy_data[invalid_indices]
            dummy_data[invalid_index]

    def test_replace_fill_values(self):
        """Tests _replace_fill_values() helper function across multiple
        different dtype arrays used as face_nodes."""

        # expected output from _replace_fill_values()
        face_nodes_gold = np.array(
            [[1, 2, self.fv], [self.fv, self.fv, self.fv]], dtype=INT_DTYPE)

        # test different datatypes for face_nodes
        dtypes = [np.int32, np.int64, np.float32, np.float64]
        for dtype in dtypes:
            # test face nodes with set dtype
            face_nodes = np.array([[1, 2, -1], [-1, -1, -1]], dtype=dtype)

            # output of _replace_fill_values()
            face_nodes_test = _replace_fill_values(grid_var=face_nodes,
                                                   original_fill=-1,
                                                   new_fill=INT_FILL_VALUE,
                                                   new_dtype=INT_DTYPE)

            assert np.array_equal(face_nodes_test, face_nodes_gold)

    def test_replace_fill_values_invalid(self):
        """Tests _replace_fill_values() helper function attempting to use a
        fill value that is not representable by the current dtype."""

        face_nodes = np.array([[1, 2, -1], [-1, -1, -1]], dtype=np.uint32)
        # invalid fill value with dtype should raise a valueError
        with self.assertRaises(ValueError):
            # INT_FILL_VALUE (max(uint32) not representable by int16)
            face_nodes_test = _replace_fill_values(grid_var=face_nodes,
                                                   original_fill=-1,
                                                   new_fill=INT_FILL_VALUE,
                                                   new_dtype=np.int16)


class TestSparseMatrix(TestCase):

    def test_convert_face_node_conn_to_sparse_matrix(self):
        """Tests _face_nodes_to_sparse_matrix() helper function to see if can
        generate sparse matrix from face_nodes_conn that has Fill Values."""
        face_nodes_conn = np.array([[3, 4, 5, INT_FILL_VALUE], [3, 0, 2, 5],
                                    [3, 4, 1, 0], [0, 1, 2, INT_FILL_VALUE]])

        face_indices, nodes_indices, non_zero_flag = ux.grid.connectivity._face_nodes_to_sparse_matrix(
            face_nodes_conn)
        expected_non_zero_flag = np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        expected_face_indices = np.array(
            [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
        expected_nodes_indices = np.array(
            [3, 4, 5, 3, 0, 2, 5, 3, 4, 1, 0, 0, 1, 2])

        nt.assert_array_equal(non_zero_flag, expected_non_zero_flag)
        nt.assert_array_equal(face_indices, expected_face_indices)
        nt.assert_array_equal(nodes_indices, expected_nodes_indices)


class TestIntersectionPoint(TestCase):

    def test_pt_within_gcr(self):

        # The GCR that's eexactly 180 degrees will have Value Error raised
        gcr_180degree_cart = [
            ux.grid.coordinates.node_lonlat_rad_to_xyz([0.0, 0.0]),
            ux.grid.coordinates.node_lonlat_rad_to_xyz([np.pi, 0.0])
        ]
        pt_same_lon_in = ux.grid.coordinates.node_lonlat_rad_to_xyz([0.0, 0.0])
        with self.assertRaises(ValueError):
            point_within_gca(pt_same_lon_in, gcr_180degree_cart)

        gcr_180degree_cart = [
            ux.grid.coordinates.node_lonlat_rad_to_xyz([0.0, np.pi / 2.0]),
            ux.grid.coordinates.node_lonlat_rad_to_xyz([0.0, -np.pi / 2.0])
        ]

        pt_same_lon_in = ux.grid.coordinates.node_lonlat_rad_to_xyz([0.0, 0.0])
        with self.assertRaises(ValueError):
            point_within_gca(pt_same_lon_in, gcr_180degree_cart)

        # Test when the point and the GCR all have the same longitude
        gcr_same_lon_cart = [
            ux.grid.coordinates.node_lonlat_rad_to_xyz([0.0, 1.5]),
            ux.grid.coordinates.node_lonlat_rad_to_xyz([0.0, -1.5])
        ]
        pt_same_lon_in = ux.grid.coordinates.node_lonlat_rad_to_xyz([0.0, 0.0])
        self.assertTrue(point_within_gca(pt_same_lon_in, gcr_same_lon_cart))

        pt_same_lon_out = ux.grid.coordinates.node_lonlat_rad_to_xyz(
            [0.0, 1.500000000000001])
        res = point_within_gca(pt_same_lon_out, gcr_same_lon_cart)
        self.assertFalse(res)

        # And if we increase the digital place by one, it should be true again
        pt_same_lon_out_add_one_place = ux.grid.coordinates.node_lonlat_rad_to_xyz(
            [0.0, 1.5000000000000001])
        res = point_within_gca(pt_same_lon_out_add_one_place, gcr_same_lon_cart)
        self.assertTrue(res)

        # Normal case
        # GCR vertex0 in radian : [1.3003315590159483, -0.007004587172323237],
        # GCR vertex1 in radian : [3.5997458123873827, -1.4893379576608758]
        # Point in radian : [1.3005410084914981, -0.010444274637648326]
        gcr_cart_2 = np.array([[0.267, 0.963, -0.007], [-0.073, -0.036,
                                                        -0.997]])
        pt_cart_within = np.array(
            [0.25616109352676675, 0.9246590335292105, -0.010021496695000144])
        self.assertTrue(point_within_gca(pt_cart_within, gcr_cart_2))

        # Test other more complicate cases : The anti-meridian case

        # GCR vertex0 in radian : [5.163808182822441, 0.6351384888657234],
        # GCR vertex1 in radian : [0.8280410325693055, 0.42237025187091526]
        # Point in radian : [0.12574759138415173, 0.770098701904903]
        gcr_cart = np.array([[0.351, -0.724, 0.593], [0.617, 0.672, 0.410]])
        pt_cart = np.array(
            [0.9438777657502077, 0.1193199333436068, 0.922714737029319])
        self.assertTrue(point_within_gca(pt_cart, gcr_cart))
        # If we swap the gcr, it should throw a value error since it's larger than 180 degree
        gcr_cart_flip = np.array([[0.617, 0.672, 0.410], [0.351, -0.724,
                                                          0.593]])
        with self.assertRaises(ValueError):
            point_within_gca(pt_cart, gcr_cart_flip)

        # 2nd anti-meridian case
        # GCR vertex0 in radian : [4.104711496596806, 0.5352983676533828],
        # GCR vertex1 in radian : [2.4269979227622533, -0.007003212877856825]
        # Point in radian : [0.43400375562899113, -0.49554509841586936]
        gcr_cart_1 = np.array([[-0.491, -0.706, 0.510], [-0.755, 0.655,
                                                         -0.007]])
        pt_cart_within = np.array(
            [0.6136726305712109, 0.28442243941920053, -0.365605190899831])
        self.assertFalse(point_within_gca(pt_cart_within, gcr_cart_1))

        # The first case should not work and the second should work
        v1_rad = [0.1, 0.0]
        v2_rad = [2 * np.pi - 0.1, 0.0]
        v1_cart = ux.grid.coordinates.node_lonlat_rad_to_xyz(v1_rad)
        v2_cart = ux.grid.coordinates.node_lonlat_rad_to_xyz(v2_rad)
        gcr_cart = np.array([v1_cart, v2_cart])
        pt_cart = ux.grid.coordinates.node_lonlat_rad_to_xyz([0.01, 0.0])
        with self.assertRaises(ValueError):
            point_within_gca(pt_cart, gcr_cart)
        gcr_car_flipped = np.array([v2_cart, v1_cart])
        self.assertTrue(point_within_gca(pt_cart, gcr_car_flipped))


class TestOperators(TestCase):

    def test_in_between(self):
        # Test the in_between operator
        self.assertTrue(in_between(0, 1, 2))
        self.assertTrue(in_between(-1, -1.5, -2))


class TestVectorsAngel(TestCase):

    def test_angle_of_2_vectors(self):
        # Test the angle between two vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(_angle_of_2_vectors(v1, v2), np.pi / 2.0)

        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(_angle_of_2_vectors(v1, v2), 0.0)
