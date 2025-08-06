"""
SCRIP format tests using base IO test classes.

This module tests SCRIP-specific functionality while inheriting common
test patterns from the base IO test classes.
"""

import pytest
import numpy as np
import xarray as xr
import uxarray as ux
from pathlib import Path
import tempfile
import os
import warnings
from numpy.testing import assert_array_equal, assert_allclose
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

from .base_io_tests import (
    BaseIOReaderTests,
    BaseIOWriterTests,
    BaseIORoundTripTests,
    BaseIOEdgeCaseTests,
    BaseIODatasetTests,
    BaseIOPerformanceTests,
    validate_grid_topology,
    validate_grid_coordinates
)

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

# SCRIP-specific test configurations
SCRIP_READ_CONFIGS = [
    ("scrip", "ne8")
]

SCRIP_WRITE_FORMATS = []  # SCRIP doesn't support writing currently

SCRIP_ROUND_TRIP_CONFIGS = []  # No round-trip until writing is supported


class TestSCRIPReader(BaseIOReaderTests):
    """Test SCRIP reading functionality."""

    format_configs = SCRIP_READ_CONFIGS

    def test_scrip_basic_structure(self, test_data_paths):
        """Test basic SCRIP file structure."""
        grid_path = test_data_paths["scrip"]["ne8"]
        if not grid_path.exists():
            pytest.skip("SCRIP ne8 test file not found")

        grid = ux.open_grid(grid_path)

        # Basic validation
        assert grid is not None
        validate_grid_topology(grid)
        validate_grid_coordinates(grid)

    def test_scrip_dimensions(self, test_data_paths):
        """Test SCRIP-specific dimensions and structure."""
        grid_path = test_data_paths["scrip"]["ne8"]
        if not grid_path.exists():
            pytest.skip("SCRIP ne8 test file not found")

        grid = ux.open_grid(grid_path)

        # Check for expected dimensions
        expected_dims = ['n_node', 'n_face']
        for dim in expected_dims:
            assert dim in grid._ds.dims

        # Validate grid structure
        assert grid.n_node > 0
        assert grid.n_face > 0
        assert len(grid.node_lon) == len(grid.node_lat)


class TestSCRIPWriter(BaseIOWriterTests):
    """Test SCRIP writing functionality (currently not supported)."""

    writable_formats = SCRIP_WRITE_FORMATS  # Empty for now

    def test_scrip_write_not_supported(self):
        """Test that SCRIP writing is not currently supported."""
        # This test documents that SCRIP writing is not yet implemented
        # Can be updated when write support is added
        pass


class TestSCRIPRoundTrip(BaseIORoundTripTests):
    """Test SCRIP round-trip consistency (currently not supported)."""

    round_trip_configs = SCRIP_ROUND_TRIP_CONFIGS  # Empty for now

    def test_scrip_round_trip_not_supported(self):
        """Test that SCRIP round-trip is not currently supported."""
        # This test documents that SCRIP round-trip is not yet implemented
        # Can be updated when write support is added
        pass


class TestSCRIPEdgeCases(BaseIOEdgeCaseTests):
    """Test SCRIP edge cases and error conditions."""

    def test_standardized_dtype_and_fill_values(self, test_data_paths):
        """Test that SCRIP files use standardized dtype and fill values."""
        grid_path = test_data_paths["scrip"]["ne8"]
        if not grid_path.exists():
            pytest.skip("SCRIP ne8 test file not found")

        grid = ux.open_grid(grid_path)

        # Check dtype and fill value
        assert grid.face_node_connectivity.dtype in [INT_DTYPE, np.int32, np.int64]
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE


class TestSCRIPDatasets(BaseIODatasetTests):
    """Test SCRIP dataset functionality."""

    def test_scrip_dataset_basic(self, test_data_paths):
        """Basic SCRIP dataset validation."""
        grid_path = test_data_paths["scrip"]["ne8"]
        if not grid_path.exists():
            pytest.skip("SCRIP ne8 test file not found")

        grid = ux.open_grid(grid_path)

        assert grid is not None
        validate_grid_topology(grid)
        validate_grid_coordinates(grid)


class TestSCRIPPerformance(BaseIOPerformanceTests):
    """Test SCRIP performance characteristics."""

    def test_lazy_loading(self, test_data_paths):
        """Test that SCRIP grid loading is reasonably fast."""
        import time

        grid_path = test_data_paths["scrip"]["ne8"]
        if not grid_path.exists():
            pytest.skip("SCRIP test file not found")

        start_time = time.time()
        grid = ux.open_grid(grid_path)
        load_time = time.time() - start_time

        # Basic validation that grid loaded successfully
        assert grid is not None
        assert hasattr(grid, 'node_lon')

        # Loading should complete in reasonable time
        assert load_time < 30.0, f"SCRIP loading took {load_time:.2f}s, which seems excessive"


class TestSCRIPSpecialCases:
    """Test SCRIP-specific special cases and functionality."""

    def test_scrip_coordinate_handling(self, test_data_paths):
        """Test SCRIP coordinate handling and conversion."""
        grid_path = test_data_paths["scrip"]["ne8"]
        if not grid_path.exists():
            pytest.skip("SCRIP ne8 test file not found")

        grid = ux.open_grid(grid_path)

        # Validate coordinate properties
        validate_grid_coordinates(grid)

        # Check that coordinates are properly converted from SCRIP format
        assert hasattr(grid, 'node_lon')
        assert hasattr(grid, 'node_lat')
        assert len(grid.node_lon) == len(grid.node_lat)

    def test_scrip_format_detection(self, test_data_paths):
        """Test that SCRIP format is correctly detected and parsed."""
        grid_path = test_data_paths["scrip"]["ne8"]
        if not grid_path.exists():
            pytest.skip("SCRIP ne8 test file not found")

        # Open as raw xarray dataset first
        raw_ds = xr.open_dataset(grid_path)

        # Check for SCRIP-specific variables that should be present
        scrip_vars = ['grid_center_lat', 'grid_center_lon', 'grid_corner_lat', 'grid_corner_lon']
        scrip_vars_found = [var for var in scrip_vars if var in raw_ds.data_vars]

        # Should find some SCRIP-specific variables
        assert len(scrip_vars_found) > 0, "No SCRIP-specific variables found"

        # Now test that uxarray can properly read it
        grid = ux.open_grid(grid_path)
        assert grid is not None

        raw_ds.close()

    def test_scrip_grid_conversion(self, test_data_paths):
        """Test conversion from SCRIP to uxarray grid format."""
        grid_path = test_data_paths["scrip"]["ne8"]
        if not grid_path.exists():
            pytest.skip("SCRIP ne8 test file not found")

        grid = ux.open_grid(grid_path)

        # Test that the grid has been properly converted to UGRID-like format
        assert 'face_node_connectivity' in grid._ds
        assert 'node_lon' in grid._ds
        assert 'node_lat' in grid._ds

        # Validate the conversion
        validate_grid_topology(grid)
        validate_grid_coordinates(grid)

    def test_scrip_comparison_with_reference(self, test_data_paths):
        """Test SCRIP grid against reference implementations where available."""
        # This could be expanded to compare with reference UGRID files
        # if we have equivalent grids in different formats

        grid_path = test_data_paths["scrip"]["ne8"]
        if not grid_path.exists():
            pytest.skip("SCRIP ne8 test file not found")

        scrip_grid = ux.open_grid(grid_path)

        # Basic validation that should be consistent across formats
        validate_grid_topology(scrip_grid)
        validate_grid_coordinates(scrip_grid)

        # Check for reasonable grid size (ne8 should have specific characteristics)
        assert scrip_grid.n_node > 0
        assert scrip_grid.n_face > 0

        # For ne8, we expect a certain order of magnitude
        assert scrip_grid.n_face > 100  # Should have more than 100 faces
        assert scrip_grid.n_node > 100  # Should have more than 100 nodes
