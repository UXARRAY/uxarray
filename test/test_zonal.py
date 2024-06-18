import numpy as np
import pandas as pd
from unittest import TestCase
from unittest.mock import patch
import numpy.testing as nt
from uxarray.core.zonal import _get_candidate_faces_at_constant_latitude, _non_conservative_zonal_mean_constant_one_latitude

class TestZonalFunctions(TestCase):

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
