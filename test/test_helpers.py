import os
import numpy as np
import numpy.testing as nt
import random
import xarray as xr
import gmpy2
from gmpy2 import mpfr, mpz
import mpmath
from unittest import TestCase
from pathlib import Path
import time

import uxarray as ux

from uxarray.helpers import _replace_fill_values, node_lonlat_rad_to_xyz, node_xyz_to_lonlat_rad
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.multi_precision_helpers import convert_to_multiprecision, set_global_precision, \
    decimal_digits_to_precision_bits

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
        face_nodes = np.array([[0, 1, 2]]).astype(INT_DTYPE)
        face_dimension = np.array([3], dtype=INT_DTYPE)
        area = ux.get_all_face_area_from_coords(x,
                                                y,
                                                z,
                                                face_nodes,
                                                face_dimension,
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
        [x, y, z] = ux.helpers.normalize_in_place(
            [random.random(), random.random(),
             random.random()])
        self.assertLessEqual(np.absolute(np.sqrt(x * x + y * y + z * z) - 1),
                             err_tolerance)

        # Multiprecision test for places=19
        precision = decimal_digits_to_precision_bits(19)
        set_global_precision(precision)
        [x_mpfr, y_mpfr,
         z_mpfr] = convert_to_multiprecision(np.array([
             '1.0000000000000000001', '0.0000000000000000009',
             '0.0000000000000000001'
         ]),
                                             precision=precision)
        normalized = ux.helpers.normalize_in_place([x_mpfr, y_mpfr, z_mpfr])
        # Calculate the sum of squares using gmpy2.fsum()
        sum_of_squares = gmpy2.fsum(
            [gmpy2.square(value) for value in normalized])
        abs = gmpy2.mul(gmpy2.reldiff(mpfr('1.0'), sum_of_squares), mpfr('1.0'))
        self.assertAlmostEqual(abs, 0, places=19)

        # Reset global precision to default
        set_global_precision()

    def test_node_xyz_to_lonlat_rad(self):
        [x, y, z] = ux.helpers.normalize_in_place([
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ])
        [lon, lat] = ux.helpers.node_xyz_to_lonlat_rad([x, y, z])
        [new_x, new_y, new_z] = ux.helpers.node_lonlat_rad_to_xyz([lon, lat])
        self.assertLessEqual(np.absolute(new_x - x), err_tolerance)
        self.assertLessEqual(np.absolute(new_y - y), err_tolerance)
        self.assertLessEqual(np.absolute(new_z - z), err_tolerance)

        # Multiprecision test for places=19
        precision = decimal_digits_to_precision_bits(19)
        set_global_precision(precision)
        # Assign 1 at the 21th decimal place, which is beyond the precision of 19 decimal places
        [x_mpfr, y_mpfr,
         z_mpfr] = convert_to_multiprecision(np.array([
             '1.000000000000000000001', '0.000000000000000000001',
             '0.000000000000000000001'
         ]),
                                             precision=precision)
        [lon_mpfr,
         lat_mpfr] = ux.helpers.node_xyz_to_lonlat_rad([x_mpfr, y_mpfr, z_mpfr])
        self.assertAlmostEqual(lon_mpfr, 0, places=19)
        self.assertAlmostEqual(lat_mpfr, 0, places=19)

        # Remove 1 at the 21th decimal place, and the total digit place are 19. the results should be perfectly equal to [0,0]
        [x_mpfr, y_mpfr,
         z_mpfr] = convert_to_multiprecision(np.array([
             '1.0000000000000000000', '0.0000000000000000000',
             '0.0000000000000000000'
         ]),
                                             precision=precision)
        [lon_mpfr,
         lat_mpfr] = ux.helpers.node_xyz_to_lonlat_rad([x_mpfr, y_mpfr, z_mpfr])
        self.assertTrue(gmpy2.cmp(lon_mpfr, mpfr('0')) == 0)
        self.assertTrue(gmpy2.cmp(lat_mpfr, mpfr('0')) == 0)

        # Reset global precision to default
        set_global_precision()

    def test_node_latlon_rad_to_xyz(self):
        [lon, lat] = [
            random.uniform(0, 2 * np.pi),
            random.uniform(-0.5 * np.pi, 0.5 * np.pi)
        ]
        [x, y, z] = ux.helpers.node_lonlat_rad_to_xyz([lon, lat])
        [new_lon, new_lat] = ux.helpers.node_xyz_to_lonlat_rad([x, y, z])
        self.assertLessEqual(np.absolute(new_lon - lon), err_tolerance)
        self.assertLessEqual(np.absolute(new_lat - lat), err_tolerance)

        # Multiprecision test for places=19
        precision = decimal_digits_to_precision_bits(19)
        set_global_precision(precision)
        # Assign 1 at the 21th decimal place, which is beyond the precision of 19 decimal places
        [lon_mpfr, lat_mpfr] = convert_to_multiprecision(np.array(
            ['0.000000000000000000001', '0.000000000000000000001']),
                                                         precision=precision)
        [x_mpfr, y_mpfr,
         z_mpfr] = ux.helpers.node_lonlat_rad_to_xyz([lon_mpfr, lat_mpfr])
        self.assertAlmostEqual(x_mpfr, 1, places=19)
        self.assertAlmostEqual(y_mpfr, 0, places=19)
        self.assertAlmostEqual(z_mpfr, 0, places=19)

        # Remove 1 at the 21th decimal place, and the total digit place are 19. the results should be perfectly equal to [1,0,0]
        [lon_mpfr, lat_mpfr] = convert_to_multiprecision(np.array(
            ['0.0000000000000000000', '0.0000000000000000000']),
                                                         precision=precision)
        [x_mpfr, y_mpfr,
         z_mpfr] = ux.helpers.node_lonlat_rad_to_xyz([lon_mpfr, lat_mpfr])
        self.assertTrue(gmpy2.cmp(x_mpfr, mpfr('1')) == 0)
        self.assertTrue(gmpy2.cmp(y_mpfr, mpfr('0')) == 0)
        self.assertTrue(gmpy2.cmp(z_mpfr, mpfr('0')) == 0)

        # Reset global precision to default
        set_global_precision()

    def test_precise_coordinates_conversion(self):
        # Multiprecision test for places=19, And we set the global precision places to 20
        # Repeat the conversion between latitude and longitude and xyz for 1000 times
        # And see if the results are the same
        precision = decimal_digits_to_precision_bits(20)
        set_global_precision(precision)

        # The initial coordinates
        [init_x, init_y, init_z] = ux.helpers.normalize_in_place([
            mpfr('0.12345678910111213149'),
            mpfr('0.92345678910111213149'),
            mpfr('1.72345678910111213149')
        ])
        new_x = init_x
        new_y = init_y
        new_z = init_z
        for iter in range(1000):
            [new_lon,
             new_lat] = ux.helpers.node_xyz_to_lonlat_rad([new_x, new_y, new_z])
            [new_x, new_y,
             new_z] = ux.helpers.node_lonlat_rad_to_xyz([new_lon, new_lat])
            self.assertAlmostEqual(new_x, init_x, places=19)
            self.assertAlmostEqual(new_y, init_y, places=19)
            self.assertAlmostEqual(new_z, init_z, places=19)

        # Test for the longitude and latitude conversion
        # The initial coordinates
        [init_lon,
         init_lat] = [mpfr('1.4000332309896247'),
                      mpfr('1.190289949682531')]
        # Reset global precision to default
        new_lat = init_lat
        new_lon = init_lon
        for iter in range(1000):
            [new_x, new_y,
             new_z] = ux.helpers.node_lonlat_rad_to_xyz([new_lon, new_lat])
            [new_lon,
             new_lat] = ux.helpers.node_xyz_to_lonlat_rad([new_x, new_y, new_z])
            self.assertAlmostEqual(new_lon, init_lon, places=19)
            self.assertAlmostEqual(new_lat, init_lat, places=19)

        # Reset global precision to default
        set_global_precision()

    def test_coordinates_conversion_accumulate_error(self):
        # Get the accumulated error of each function call
        ux.multi_precision_helpers.set_global_precision(64)
        run_time = 100
        print("\n")

        # Using the float number
        new_lon = 122.987654321098765
        new_lat = 36.123456789012345

        start_time = time.time()
        for iter in range(run_time):
            [new_x, new_y, new_z
            ] = ux.helpers.node_lonlat_rad_to_xyz(np.deg2rad([new_lon,
                                                              new_lat]))
            [new_lon,
             new_lat] = ux.helpers.node_xyz_to_lonlat_rad([new_x, new_y, new_z])
            [new_lon, new_lat] = np.rad2deg([new_lon, new_lat])

        end_time = time.time()
        diff_lat = mpfr(str(new_lat)) - mpfr('36.123456789012345')
        diff_lon = mpfr(str(new_lon)) - mpfr('122.987654321098765')
        print("The floating point longitude accumulated error is: " +
              str(diff_lon) + "and the latitude accumulated "
              "error is: " + str(diff_lat))
        print("The floating point Execution time: ", end_time - start_time,
              " seconds")

        # Get the accumulated error of each function call
        print("\n")

        # Using the float number

        [init_lon, init_lat
        ] = [mpfr('122.987654321098765', 64),
             mpfr('36.123456789012345', 64)]
        new_lon = mpfr('122.987654321098765', 64)
        new_lat = mpfr('36.123456789012345', 64)
        start_time = time.time()

        for iter in range(run_time):
            [new_x, new_y, new_z] = ux.helpers.node_lonlat_rad_to_xyz(
                [gmpy2.radians(val) for val in [new_lon, new_lat]])
            [new_lon,
             new_lat] = ux.helpers.node_xyz_to_lonlat_rad([new_x, new_y, new_z])
            [new_lon,
             new_lat] = [gmpy2.degrees(val) for val in [new_lon, new_lat]]
        end_time = time.time()
        diff_lat = new_lat - init_lat
        diff_lon = new_lon - init_lon
        print("The mpfr longitude accumulated error is: " + str(diff_lon) +
              " and the latitude accumulated error is: " + str(diff_lat))
        print("The mpfr Execution time: ", end_time - start_time, " seconds")


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


class TestIntersectionPoint(TestCase):

    def test_pt_within_gcr(self):

        # Test when the point and the GCR all have the same longitude
        gcr_same_lon_cart = [
            ux.helpers.node_lonlat_rad_to_xyz([0, 1.5]),
            ux.helpers.node_lonlat_rad_to_xyz([0, -1.5])
        ]
        pt_same_lon_in = ux.helpers.node_lonlat_rad_to_xyz([0, 0])
        self.assertTrue(
            ux.helpers.point_within_GCR(pt_same_lon_in, gcr_same_lon_cart))

        pt_same_lon_out = ux.helpers.node_lonlat_rad_to_xyz(
            [0, 1.500000000000005])
        res = ux.helpers.point_within_GCR(pt_same_lon_out, gcr_same_lon_cart)
        self.assertFalse(res)

        # And if we increase the digital place by one, it should be true again
        pt_same_lon_out_add_one_place = ux.helpers.node_lonlat_rad_to_xyz(
            [0, 1.5000000000000005])
        res = ux.helpers.point_within_GCR(pt_same_lon_out_add_one_place, gcr_same_lon_cart)
        self.assertTrue(res)

        # Normal case
        # GCR vertex0 in radian : [1.3003315590159483, -0.007004587172323237],
        # GCR vertex1 in radian : [3.5997458123873827, -1.4893379576608758]
        # Point in radian : [1.3005410084914981, -0.010444274637648326]
        gcr_cart_2 = np.array([[0.267, 0.963, -0.007], [-0.073, -0.036,
                                                        -0.997]])
        pt_cart_within = np.array(
            [0.25616109352676675, 0.9246590335292105, -0.010021496695000144])
        self.assertTrue(ux.helpers.point_within_GCR(pt_cart_within, gcr_cart_2))

        # Test other more complicate cases : The anti-meridian case

        # GCR vertex0 in radian : [5.163808182822441, 0.6351384888657234],
        # GCR vertex1 in radian : [0.8280410325693055, 0.42237025187091526]
        # Point in radian : [0.12574759138415173, 0.770098701904903]
        gcr_cart = np.array([[0.351, -0.724, 0.593], [0.617, 0.672, 0.410]])
        pt_cart = np.array(
            [0.9438777657502077, 0.1193199333436068, 0.922714737029319])
        self.assertTrue(ux.helpers.point_within_GCR(pt_cart, gcr_cart))
        # If we swap the gcr, it should still be true
        gcr_cart_flip = np.array([[0.617, 0.672, 0.410], [0.351, -0.724,
                                                          0.593]])
        self.assertTrue(ux.helpers.point_within_GCR(pt_cart, gcr_cart_flip))

        # 2nd anti-meridian case
        # GCR vertex0 in radian : [4.104711496596806, 0.5352983676533828],
        # GCR vertex1 in radian : [2.4269979227622533, -0.007003212877856825]
        # Point in radian : [0.43400375562899113, -0.49554509841586936]
        gcr_cart_1 = np.array([[-0.491, -0.706, 0.510], [-0.755, 0.655,
                                                         -0.007]])
        pt_cart_within = np.array(
            [0.6136726305712109, 0.28442243941920053, -0.365605190899831])
        self.assertFalse(ux.helpers.point_within_GCR(pt_cart_within, gcr_cart_1))

        # The following two case should work even swapping the GCR
        v1_rad = [0.1, 0]
        v2_rad = [2 * np.pi - 0.1, 0]
        v1_cart = ux.helpers.node_lonlat_rad_to_xyz(v1_rad)
        v2_cart = ux.helpers.node_lonlat_rad_to_xyz(v2_rad)
        gcr_cart = np.array([v1_cart, v2_cart])
        pt_cart = ux.helpers.node_lonlat_rad_to_xyz([0.01, 0])
        self.assertTrue(ux.helpers.point_within_GCR(pt_cart, gcr_cart))
        gcr_car_flipped = np.array([v2_cart, v1_cart])
        self.assertTrue(ux.helpers.point_within_GCR(pt_cart, gcr_car_flipped))

    def test_pt_within_gcr_multiprecision(self):

        set_global_precision(55)

        # Test when the point and the GCR all have the same longitude
        gcr_same_lon_cart = [
            ux.helpers.node_lonlat_rad_to_xyz([mpfr('0'),
                                               mpfr('1.5')]),
            ux.helpers.node_lonlat_rad_to_xyz([mpfr('0'),
                                               mpfr('-1.5')])
        ]
        pt_same_lon_in = ux.helpers.node_lonlat_rad_to_xyz(
            [mpfr('0'), mpfr('0')])
        self.assertTrue(
            ux.helpers.point_within_GCR(pt_same_lon_in, gcr_same_lon_cart))

        pt_same_lon_out = ux.helpers.node_lonlat_rad_to_xyz(
            [mpfr('0'), mpfr('1.500000000000005')])
        res = ux.helpers.point_within_GCR(pt_same_lon_out, gcr_same_lon_cart)
        self.assertFalse(res)

        # And if we increase the digital place by one, it should be still be false in the multiprecision case
        pt_same_lon_out_add_one_place = ux.helpers.node_lonlat_rad_to_xyz(
            [mpfr('0'), mpfr('1.5000000000000005')])
        res = ux.helpers.point_within_GCR(pt_same_lon_out_add_one_place, gcr_same_lon_cart)
        self.assertFalse(res)

        # Normal case
        GCR_cart = np.array([
            node_lonlat_rad_to_xyz([mpfr('3.14'), mpfr('0')]),
            node_lonlat_rad_to_xyz([mpfr('6.28'), mpfr('0')])
        ])
        pt_cart_within = np.array(
            node_lonlat_rad_to_xyz([mpfr('6.27999999999999999'),
                                    mpfr('0')]))
        res = ux.helpers.point_within_GCR(pt_cart_within, GCR_cart)
        self.assertTrue(res)

        # Test other more complicate cases : The anti-meridian case
        GCR_cart = np.array([
            node_lonlat_rad_to_xyz([mpfr('6.0'), mpfr('0')]),
            node_lonlat_rad_to_xyz([mpfr('1.0'), mpfr('0')])
        ])
        pt_cart_within = np.array(
            node_lonlat_rad_to_xyz([mpfr('6.00000000000000001'),
                                    mpfr('0')]))
        res = ux.helpers.point_within_GCR(pt_cart_within, GCR_cart)
        self.assertTrue(res)

        GCR_cart = np.array([
            node_lonlat_rad_to_xyz([mpfr('1.0'), mpfr('0')]),
            node_lonlat_rad_to_xyz([mpfr('6.0'), mpfr('0')])
        ])
        pt_cart_within = np.array(
            node_lonlat_rad_to_xyz([mpfr('5.999999999999999'),
                                    mpfr('0')]))
        res = ux.helpers.point_within_GCR(pt_cart_within, GCR_cart)
        self.assertFalse(res)

        # The following two case should work even swapping the GCR
        v1_rad = [mpfr('0.1'), mpfr('0')]
        v2_rad = [
            mpfr('2') * gmpy2.const_pi() - mpfr('0.000000000000001'),
            mpfr('0')
        ]
        v1_cart = ux.helpers.node_lonlat_rad_to_xyz(v1_rad)
        v2_cart = ux.helpers.node_lonlat_rad_to_xyz(v2_rad)
        gcr_cart = np.array([v1_cart, v2_cart])
        pt_cart = ux.helpers.node_lonlat_rad_to_xyz(
            [mpfr('0.00000000000000001'),
             mpfr('0')])
        self.assertTrue(ux.helpers.point_within_GCR(pt_cart, gcr_cart))
        gcr_car_flipped = np.array([v2_cart, v1_cart])
        self.assertTrue(ux.helpers.point_within_GCR(pt_cart, gcr_car_flipped))

        set_global_precision()
