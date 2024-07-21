import numpy as np
import pandas as pd
import uxarray as ux
from unittest import TestCase
from unittest.mock import patch
import numpy.testing as nt
from uxarray.core.zonal import _get_candidate_faces_at_constant_latitude, _non_conservative_zonal_mean_constant_one_latitude
import os
from pathlib import Path

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

class TestZonalFunctions(TestCase):
    gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
    datafile_vortex_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"
    dsfile_var2_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_var2.nc"
    test_file_2 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_test2.nc"
    test_file_3 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_test3.nc"

    def test_get_candidate_faces_at_constant_latitude(self):
        """Test _get_candidate_faces_at_constant_latitude function."""

        # Create test data
        bounds = np.array([
            [[-45, 45], [0, 360]],
            [[-90, -45], [0, 360]],
            [[45, 90], [0, 360]],
        ])
        constLat = 0

        # Get candidate faces
        candidate_faces = _get_candidate_faces_at_constant_latitude(bounds, constLat)

        # Expected output
        expected_faces = np.array([0])

        # Test the function output
        nt.assert_array_equal(candidate_faces, expected_faces)

    def test_get_candidate_faces_at_constant_latitude_out_of_bounds(self):
        """Test _get_candidate_faces_at_constant_latitude with out-of-bounds
        latitude."""

        # Create test data
        bounds = np.array([
            [[-45, 45], [0, 360]],
            [[-90, -45], [0, 360]],
            [[45, 90], [0, 360]],
        ])
        constLat = 100  # Out of bounds

        # Test for ValueError
        with self.assertRaises(ValueError):
            _get_candidate_faces_at_constant_latitude(bounds, constLat)

    @patch('uxarray.core.zonal._get_zonal_faces_weight_at_constLat', return_value=pd.DataFrame({
                                                                                                    'face_index': [0],
                                                                                                    'weight': [0.2]
                                                                                                }))
    def test_non_conservative_zonal_mean_constant_one_latitude_one_candidate(self, mock_get_zonal_faces_weight_at_constLat):
        """Test _non_conservative_zonal_mean_constant_one_latitude function."""

        # Create test data
        face_edges_cart = np.random.rand(3, 4, 2, 3)
        face_bounds = np.array([
            [[-45, 45], [0, 360]],
            [[-90, -45], [0, 360]],
            [[45, 90], [0, 360]],
        ])
        face_data = np.array([1.0, 2.0, 3.0])
        constLat = 0

        # Get zonal mean
        zonal_mean = _non_conservative_zonal_mean_constant_one_latitude(
            face_edges_cart, face_bounds, face_data, constLat)

        # Expected output
        expected_zonal_mean = 1

        # Test the function output
        nt.assert_almost_equal(zonal_mean, expected_zonal_mean)

    @patch('uxarray.core.zonal._get_zonal_faces_weight_at_constLat', return_value=pd.DataFrame({
                                                                                                    'face_index': [0, 1],
                                                                                                    'weight': [0.2, 0.3]
                                                                                                }))
    def test_non_conservative_zonal_mean_constant_one_latitude_two_candidate(self, mock_get_zonal_faces_weight_at_constLat):
        """Test _non_conservative_zonal_mean_constant_one_latitude function."""

        # Create test data
        face_edges_cart = np.random.rand(3, 4, 2, 3)
        face_bounds = np.array([
            [[-45, 45], [0, 360]],
            [[-90, 45], [0, 360]],
            [[45, 90], [0, 360]],
        ])
        face_data = np.array([1.0, 2.0, 3.0])
        constLat = 0

        # Get zonal mean
        zonal_mean = _non_conservative_zonal_mean_constant_one_latitude(
            face_edges_cart, face_bounds, face_data, constLat)

        # Expected output
        expected_zonal_mean = (1.0 * 0.2 + 2.0 * 0.3) / (0.2 + 0.3)

        # Test the function output
        nt.assert_almost_equal(zonal_mean, expected_zonal_mean)

    @patch('uxarray.core.zonal._get_zonal_faces_weight_at_constLat', return_value=pd.DataFrame({
                                                                                                    'face_index': [0, 1, 2],
                                                                                                    'weight': [0.2, 0.3, 0.5]
                                                                                                }))
    def test_non_conservative_zonal_mean_constant_one_latitude_all_faces(self, mock_get_zonal_faces_weight_at_constLat):
        """Test _non_conservative_zonal_mean_constant_one_latitude function."""

        # Create test data
        face_edges_cart = np.random.rand(3, 4, 2, 3)
        face_bounds = np.array([
            [[-45, 45], [0, 360]],
            [[-90, 45], [0, 360]],
            [[-45, 90], [0, 360]],
        ])
        face_data = np.array([1.0, 2.0, 3.0])
        constLat = 0

        # Get zonal mean
        zonal_mean = _non_conservative_zonal_mean_constant_one_latitude(
            face_edges_cart, face_bounds, face_data, constLat)

        # Expected output
        expected_zonal_mean = (1.0 * 0.2 + 2.0 * 0.3 + 3.0 * 0.5) / (0.2 + 0.3 + 0.5)

        # Test the function output
        nt.assert_almost_equal(zonal_mean, expected_zonal_mean)

    def test_non_conservative_zonal_mean_constant_one_latitude_no_candidate(self):
        """Test _non_conservative_zonal_mean_constant_one_latitude with no
        candidate faces."""

        # Create test data
        face_edges_cart = np.random.rand(3, 4, 2, 3)
        face_bounds = np.array([
            [[-45, -30], [0, 360]],  # Bounds that don't include the latitude 0
            [[-90, -45], [0, 360]],
            [[30, 90], [0, 360]],
        ])
        face_data = np.array([1.0, 2.0, 3.0])
        constLat = 0

        # Get zonal mean
        zonal_mean = _non_conservative_zonal_mean_constant_one_latitude(face_edges_cart, face_bounds, face_data, constLat)

        # Expected output is NaN
        self.assertTrue(np.isnan(zonal_mean))

    def test_non_conservative_zonal_mean_outCSne30(self):
        """Test _non_conservative_zonal_mean function with outCSne30 data.

        Dummy test to make sure the function runs without errors.
        """
        # Create test data
        grid_path = self.gridfile_ne30
        data_path = self.datafile_vortex_ne30
        uxds = ux.open_dataset(grid_path, data_path)

        res = uxds['psi'].zonal_mean((-90,90,1))
        print(res)

    def test_non_conservative_zonal_mean_outCSne30_test2(self):
        # Create test data
        grid_path = self.gridfile_ne30
        data_path = self.test_file_2
        uxds = ux.open_dataset(grid_path, data_path)
        res = uxds['Psi'].zonal_mean((-1.57,1.57,0.5))
        # test the output is within 1 of 2
        self.assertAlmostEqual(res.values, 2, delta=1)
        res_0 = uxds['Psi'].zonal_mean(0)
        # test the output is within 1 of 2
        self.assertAlmostEqual(res_0.values, 2, delta=1)

    def test_non_conservative_zonal_mean_outCSne30_test3(self):
        # Create test data
        grid_path = self.gridfile_ne30
        data_path = self.test_file_3
        uxds = ux.open_dataset(grid_path, data_path)
        res = uxds['Psi'].zonal_mean((-1.57,1.57,0.5))
        # test the output is within 1 of 2
        self.assertAlmostEqual(res.values, 2, delta=1)
        res_0 = uxds['Psi'].zonal_mean(0)
        # test the output is within 1 of 2
        self.assertAlmostEqual(res_0.values, 2, delta=1)
