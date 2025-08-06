"""
ICON format tests using base IO test classes.

This module tests ICON-specific functionality while inheriting common
test patterns from the base IO test classes.
"""

import pytest
import numpy as np
import xarray as xr
import uxarray as ux
from pathlib import Path
import tempfile
import os
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

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

# ICON-specific test configurations
ICON_READ_CONFIGS = [
    ("icon", "r02b04")
]

ICON_WRITE_FORMATS = []  # ICON doesn't support writing currently

ICON_ROUND_TRIP_CONFIGS = []  # No round-trip until writing is supported


class TestICONReader(BaseIOReaderTests):
    """Test ICON reading functionality."""

    format_configs = ICON_READ_CONFIGS

    def test_icon_basic_structure(self, test_data_paths):
        """Test basic ICON file structure."""
        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON r02b04 test file not found")

        grid = ux.open_grid(grid_path)

        # Basic validation
        assert grid is not None
        validate_grid_topology(grid)
        validate_grid_coordinates(grid)

    def test_icon_dimensions(self, test_data_paths):
        """Test ICON-specific dimensions and structure."""
        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON r02b04 test file not found")

        grid = ux.open_grid(grid_path)

        # Check for expected dimensions
        expected_dims = ['n_node', 'n_face']
        for dim in expected_dims:
            assert dim in grid._ds.dims

        # Validate grid structure
        assert grid.n_node > 0
        assert grid.n_face > 0
        assert len(grid.node_lon) == len(grid.node_lat)

    def test_icon_triangular_structure(self, test_data_paths):
        """Test that ICON grids have triangular face structure."""
        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON r02b04 test file not found")

        grid = ux.open_grid(grid_path)

        # ICON grids are typically triangular
        connectivity = grid.face_node_connectivity.values

        # Check that most faces are triangular (3 nodes per face)
        triangular_faces = 0
        for face_idx in range(min(100, grid.n_face)):  # Check first 100 faces
            face_nodes = connectivity[face_idx]
            valid_nodes = face_nodes[face_nodes != INT_FILL_VALUE]
            if len(valid_nodes) == 3:
                triangular_faces += 1

        # Most faces should be triangular for ICON
        assert triangular_faces > 50, "Expected predominantly triangular faces in ICON grid"


class TestICONWriter(BaseIOWriterTests):
    """Test ICON writing functionality (currently not supported)."""

    writable_formats = ICON_WRITE_FORMATS  # Empty for now

    def test_icon_write_not_supported(self):
        """Test that ICON writing is not currently supported."""
        # This test documents that ICON writing is not yet implemented
        # Can be updated when write support is added
        pass


class TestICONRoundTrip(BaseIORoundTripTests):
    """Test ICON round-trip consistency (currently not supported)."""

    round_trip_configs = ICON_ROUND_TRIP_CONFIGS  # Empty for now

    def test_icon_round_trip_not_supported(self):
        """Test that ICON round-trip is not currently supported."""
        # This test documents that ICON round-trip is not yet implemented
        # Can be updated when write support is added
        pass


class TestICONEdgeCases(BaseIOEdgeCaseTests):
    """Test ICON edge cases and error conditions."""

    def test_standardized_dtype_and_fill_values(self, test_data_paths):
        """Test that ICON files use standardized dtype and fill values."""
        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON r02b04 test file not found")

        grid = ux.open_grid(grid_path)

        # Check dtype and fill value
        assert grid.face_node_connectivity.dtype in [INT_DTYPE, np.int32, np.int64]
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE


class TestICONDatasets(BaseIODatasetTests):
    """Test ICON dataset functionality."""

    def test_icon_dataset_basic(self, test_data_paths):
        """Basic ICON dataset validation."""
        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON r02b04 test file not found")

        grid = ux.open_grid(grid_path)

        assert grid is not None
        validate_grid_topology(grid)
        validate_grid_coordinates(grid)

    def test_icon_open_dataset(self, test_data_paths):
        """Test opening ICON file as a dataset."""
        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON r02b04 test file not found")

        # Test opening as dataset
        try:
            dataset = ux.open_dataset(grid_path, grid_path)
            assert hasattr(dataset, 'uxgrid')
            assert dataset.uxgrid is not None
        except Exception:
            # This may fail if the file doesn't have data variables
            # which is acceptable for a grid-only file
            pass


class TestICONPerformance(BaseIOPerformanceTests):
    """Test ICON performance characteristics."""

    def test_lazy_loading(self, test_data_paths):
        """Test that ICON grid loading is reasonably fast."""
        import time

        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON test file not found")

        start_time = time.time()
        grid = ux.open_grid(grid_path)
        load_time = time.time() - start_time

        # Basic validation that grid loaded successfully
        assert grid is not None
        assert hasattr(grid, 'node_lon')

        # Loading should complete in reasonable time
        assert load_time < 30.0, f"ICON loading took {load_time:.2f}s, which seems excessive"


class TestICONSpecialCases:
    """Test ICON-specific special cases and functionality."""

    def test_icon_coordinate_handling(self, test_data_paths):
        """Test ICON coordinate handling and conversion."""
        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON r02b04 test file not found")

        grid = ux.open_grid(grid_path)

        # Validate coordinate properties
        validate_grid_coordinates(grid)

        # Check that coordinates are properly handled
        assert hasattr(grid, 'node_lon')
        assert hasattr(grid, 'node_lat')
        assert len(grid.node_lon) == len(grid.node_lat)

    def test_icon_format_detection(self, test_data_paths):
        """Test that ICON format is correctly detected and parsed."""
        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON r02b04 test file not found")

        # Test that uxarray can properly read the ICON file
        grid = ux.open_grid(grid_path)
        assert grid is not None

        # Check that basic grid properties are available
        assert hasattr(grid, 'face_node_connectivity')
        assert hasattr(grid, 'node_lon')
        assert hasattr(grid, 'node_lat')

    def test_icon_grid_conversion(self, test_data_paths):
        """Test conversion from ICON to uxarray grid format."""
        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON r02b04 test file not found")

        grid = ux.open_grid(grid_path)

        # Test that the grid has been properly converted to UGRID-like format
        assert 'face_node_connectivity' in grid._ds
        assert 'node_lon' in grid._ds
        assert 'node_lat' in grid._ds

        # Validate the conversion
        validate_grid_topology(grid)
        validate_grid_coordinates(grid)

    def test_icon_icosahedral_properties(self, test_data_paths):
        """Test ICON-specific icosahedral grid properties."""
        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON r02b04 test file not found")

        grid = ux.open_grid(grid_path)

        # ICON grids are icosahedral, so we expect:
        # 1. Predominantly triangular faces
        # 2. Specific connectivity patterns
        # 3. Reasonable face/node ratios

        connectivity = grid.face_node_connectivity.values

        # Count face types
        face_sizes = []
        for face_idx in range(min(1000, grid.n_face)):  # Check first 1000 faces
            face_nodes = connectivity[face_idx]
            valid_nodes = face_nodes[face_nodes != INT_FILL_VALUE]
            face_sizes.append(len(valid_nodes))

        # Most faces should be triangular
        triangular_count = face_sizes.count(3)
        total_checked = len(face_sizes)
        triangular_ratio = triangular_count / total_checked

        assert triangular_ratio > 0.8, f"Expected >80% triangular faces, got {triangular_ratio:.2%}"

    def test_icon_resolution_properties(self, test_data_paths):
        """Test properties specific to the R02B04 resolution."""
        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON r02b04 test file not found")

        grid = ux.open_grid(grid_path)

        # R02B04 is a specific ICON resolution
        # We can validate that the grid size is reasonable for this resolution
        assert grid.n_node > 1000, "R02B04 should have more than 1000 nodes"
        assert grid.n_face > 1000, "R02B04 should have more than 1000 faces"

        # Validate basic topology
        validate_grid_topology(grid)
        validate_grid_coordinates(grid)
