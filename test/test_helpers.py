import os
import numpy as np
import numpy.testing as nt
import random
import xarray as xr

from unittest import TestCase
from pathlib import Path

import uxarray as ux

from uxarray.utils.helpers import _replace_fill_values
from uxarray.utils.constants import INT_DTYPE, INT_FILL_VALUE

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
        ds_scrip_CSne8 = xr.open_dataset(gridfile_scrip_CSne8)

        # select actual center_lat/lon
        scrip_center_lon = ds_scrip_CSne8['grid_center_lon']
        scrip_center_lat = ds_scrip_CSne8['grid_center_lat']

        # Calculate the center_lat/lon using same dataset's corner_lat/lon
        calc_center = ux.grid_center_lat_lon(ds_scrip_CSne8)
        calc_lat = calc_center[0]
        calc_lon = calc_center[1]

        # Test that calculated center_lat/lon is the same as actual center_lat/lon
        np.testing.assert_array_almost_equal(scrip_center_lat, calc_lat)
        np.testing.assert_array_almost_equal(scrip_center_lon, calc_lon)


class TestCoordinatesConversion(TestCase):

    def test_normalize_in_place(self):
        [x, y, z] = ux.utils.helpers.normalize_in_place(
            [random.random(), random.random(),
             random.random()])

        self.assertLessEqual(np.absolute(np.sqrt(x * x + y * y + z * z) - 1),
                             err_tolerance)

    def test_node_xyz_to_lonlat_rad(self):
        [x, y, z] = ux.utils.helpers.normalize_in_place([
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        ])

        [lon, lat] = ux.utils.helpers.node_xyz_to_lonlat_rad([x, y, z])
        [new_x, new_y,
         new_z] = ux.utils.helpers.node_lonlat_rad_to_xyz([lon, lat])

        self.assertLessEqual(np.absolute(new_x - x), err_tolerance)
        self.assertLessEqual(np.absolute(new_y - y), err_tolerance)
        self.assertLessEqual(np.absolute(new_z - z), err_tolerance)

    def test_node_latlon_rad_to_xyz(self):
        [lon, lat] = [
            random.uniform(0, 2 * np.pi),
            random.uniform(-0.5 * np.pi, 0.5 * np.pi)
        ]

        [x, y, z] = ux.utils.helpers.node_lonlat_rad_to_xyz([lon, lat])

        [new_lon, new_lat] = ux.utils.helpers.node_xyz_to_lonlat_rad([x, y, z])

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
