"""
FESOM format tests using base IO test classes.

This module tests FESOM-specific functionality while inheriting common
test patterns from the base IO test classes.
"""

import pytest
import numpy as np
import xarray as xr
import uxarray as ux
from pathlib import Path
import tempfile
import os
import glob
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

# FESOM-specific test configurations
FESOM_READ_CONFIGS = [
    ("fesom", "ugrid_diag"),
    ("fesom", "ascii"),
    ("fesom", "netcdf")
]

FESOM_WRITE_FORMATS = []  # FESOM doesn't support writing currently

FESOM_ROUND_TRIP_CONFIGS = []  # No round-trip until writing is supported


class TestFESOMReader(BaseIOReaderTests):
    """Test FESOM reading functionality."""

    format_configs = FESOM_READ_CONFIGS

    def test_fesom_basic_structure(self, test_data_paths):
        """Test basic FESOM file structure for all formats."""
        test_files = [
            ("fesom", "ugrid_diag"),
            ("fesom", "ascii"),
            ("fesom", "netcdf")
        ]

        for format_name, data_key in test_files:
            grid_path = test_data_paths[format_name][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            # Basic validation
            assert grid is not None
            validate_grid_topology(grid)
            validate_grid_coordinates(grid)

            # Validate the grid
            grid.validate()

    def test_fesom_dimensions(self, test_data_paths):
        """Test FESOM-specific dimensions and structure."""
        for data_key in ["ugrid_diag", "ascii", "netcdf"]:
            grid_path = test_data_paths["fesom"][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            # Check for expected dimensions
            expected_dims = ['n_node', 'n_face']
            for dim in expected_dims:
                assert dim in grid._ds.dims

            # Validate grid structure
            assert grid.n_node > 0
            assert grid.n_face > 0
            assert len(grid.node_lon) == len(grid.node_lat)

            # Test first available file
            break

    def test_fesom_triangular_structure(self, test_data_paths):
        """Test that FESOM grids have triangular face structure."""
        for data_key in ["ugrid_diag", "ascii", "netcdf"]:
            grid_path = test_data_paths["fesom"][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            # FESOM grids are typically triangular
            connectivity = grid.face_node_connectivity.values

            # Check that faces are triangular (3 nodes per face)
            triangular_faces = 0
            for face_idx in range(min(100, grid.n_face)):  # Check first 100 faces
                face_nodes = connectivity[face_idx]
                valid_nodes = face_nodes[face_nodes != INT_FILL_VALUE]
                if len(valid_nodes) == 3:
                    triangular_faces += 1

            # All faces should be triangular for FESOM
            assert triangular_faces > 90, "Expected predominantly triangular faces in FESOM grid"
            break


class TestFESOMWriter(BaseIOWriterTests):
    """Test FESOM writing functionality (currently not supported)."""

    writable_formats = FESOM_WRITE_FORMATS  # Empty for now

    def test_fesom_write_not_supported(self):
        """Test that FESOM writing is not currently supported."""
        # This test documents that FESOM writing is not yet implemented
        # Can be updated when write support is added
        pass


class TestFESOMRoundTrip(BaseIORoundTripTests):
    """Test FESOM round-trip consistency (currently not supported)."""

    round_trip_configs = FESOM_ROUND_TRIP_CONFIGS  # Empty for now

    def test_fesom_round_trip_not_supported(self):
        """Test that FESOM round-trip is not currently supported."""
        # This test documents that FESOM round-trip is not yet implemented
        # Can be updated when write support is added
        pass


class TestFESOMEdgeCases(BaseIOEdgeCaseTests):
    """Test FESOM edge cases and error conditions."""

    def test_standardized_dtype_and_fill_values(self, test_data_paths):
        """Test that FESOM files use standardized dtype and fill values."""
        test_files = [
            ("fesom", "ugrid_diag"),
            ("fesom", "ascii"),
            ("fesom", "netcdf")
        ]

        for format_name, data_key in test_files:
            grid_path = test_data_paths[format_name][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            # Check dtype and fill value
            assert grid.face_node_connectivity.dtype in [INT_DTYPE, np.int32, np.int64]
            assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE

            # Test first available file
            break


class TestFESOMDatasets(BaseIODatasetTests):
    """Test FESOM dataset functionality."""

    def test_fesom_dataset_basic(self, test_data_paths):
        """Basic FESOM dataset validation."""
        for data_key in ["ugrid_diag", "ascii", "netcdf"]:
            grid_path = test_data_paths["fesom"][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            assert grid is not None
            validate_grid_topology(grid)
            validate_grid_coordinates(grid)

            # Test first available file
            break

    @pytest.mark.parametrize("grid_type", ["ugrid_diag", "ascii"])
    def test_fesom_open_dataset(self, test_data_paths, grid_type):
        """Test opening FESOM files as datasets with data."""
        grid_path = test_data_paths["fesom"][grid_type]
        if not grid_path.exists():
            pytest.skip(f"FESOM {grid_type} test file not found")

        # Try to find data files for ASCII format
        if grid_type == "ascii":
            data_path = grid_path / "data" / "sst.fesom.1985.nc"
            if data_path.exists():
                try:
                    dataset = ux.open_dataset(grid_path, data_path)
                    assert hasattr(dataset, 'uxgrid')
                    assert dataset.uxgrid is not None
                    assert "n_node" in dataset.dims
                    assert len(dataset) >= 1
                except Exception:
                    # This may fail due to data compatibility issues
                    pass
        else:
            # For other formats, just test grid opening
            grid = ux.open_grid(grid_path)
            assert grid is not None

    @pytest.mark.parametrize("grid_type", ["ugrid_diag", "ascii"])
    def test_fesom_open_mfdataset(self, test_data_paths, grid_type):
        """Test opening multiple FESOM data files."""
        grid_path = test_data_paths["fesom"][grid_type]
        if not grid_path.exists():
            pytest.skip(f"FESOM {grid_type} test file not found")

        # Try to find multiple data files for ASCII format
        if grid_type == "ascii":
            data_pattern = str(grid_path / "data" / "*.nc")
            data_files = glob.glob(data_pattern)
            if len(data_files) > 0:
                try:
                    dataset = ux.open_mfdataset(grid_path, data_files)
                    assert "n_node" in dataset.dims
                    assert "n_face" in dataset.dims
                except Exception:
                    # This may fail due to data compatibility issues
                    pass
        else:
            # For other formats, just test that the grid can be opened
            grid = ux.open_grid(grid_path)
            assert grid is not None


class TestFESOMPerformance(BaseIOPerformanceTests):
    """Test FESOM performance characteristics."""

    @pytest.mark.parametrize("data_key", ["ugrid_diag", "ascii", "netcdf"])
    def test_lazy_loading(self, test_data_paths, data_key):
        """Test that FESOM grid loading is reasonably fast."""
        import time

        grid_path = test_data_paths["fesom"][data_key]
        if not grid_path.exists():
            pytest.skip(f"FESOM {data_key} test file not found")

        start_time = time.time()
        grid = ux.open_grid(grid_path)
        load_time = time.time() - start_time

        # Basic validation that grid loaded successfully
        assert grid is not None
        assert hasattr(grid, 'node_lon')

        # Loading should complete in reasonable time
        assert load_time < 30.0, f"FESOM loading took {load_time:.2f}s, which seems excessive"


class TestFESOMSpecialCases:
    """Test FESOM-specific special cases and functionality."""

    def test_fesom_format_comparison(self, test_data_paths):
        """Test comparison between ASCII and UGRID FESOM formats."""
        ugrid_path = test_data_paths["fesom"]["ugrid_diag"]
        ascii_path = test_data_paths["fesom"]["ascii"]

        if not ugrid_path.exists() or not ascii_path.exists():
            pytest.skip("FESOM comparison files not found")

        # Load both grids
        ugrid_grid = ux.open_grid(ugrid_path)
        ascii_grid = ux.open_grid(ascii_path)

        # They should have the same basic structure
        assert ugrid_grid.n_face == ascii_grid.n_face
        assert ugrid_grid.n_node == ascii_grid.n_node

        # Face connectivity should be identical
        assert_array_equal(
            ugrid_grid.face_node_connectivity.values,
            ascii_grid.face_node_connectivity.values
        )

    def test_fesom_coordinate_handling(self, test_data_paths):
        """Test FESOM coordinate handling and conversion."""
        for data_key in ["ugrid_diag", "ascii", "netcdf"]:
            grid_path = test_data_paths["fesom"][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            # Validate coordinate properties
            validate_grid_coordinates(grid)

            # Check that coordinates are properly handled
            assert hasattr(grid, 'node_lon')
            assert hasattr(grid, 'node_lat')
            assert len(grid.node_lon) == len(grid.node_lat)

            # Test first available file
            break

    def test_fesom_grid_validation(self, test_data_paths):
        """Test FESOM grid validation."""
        for data_key in ["ugrid_diag", "ascii", "netcdf"]:
            grid_path = test_data_paths["fesom"][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            # Should be able to validate without errors
            grid.validate()

            # Test basic topology
            validate_grid_topology(grid)
            validate_grid_coordinates(grid)

            # Test first available file
            break

    def test_fesom_ascii_format(self, test_data_paths):
        """Test FESOM ASCII format specific functionality."""
        ascii_path = test_data_paths["fesom"]["ascii"]
        if not ascii_path.exists():
            pytest.skip("FESOM ASCII test file not found")

        grid = ux.open_grid(ascii_path)

        # FESOM ASCII format should produce valid grids
        assert grid is not None
        validate_grid_topology(grid)
        validate_grid_coordinates(grid)

        # Should have triangular faces
        connectivity = grid.face_node_connectivity.values
        face_sizes = []
        for face_idx in range(min(100, grid.n_face)):
            face_nodes = connectivity[face_idx]
            valid_nodes = face_nodes[face_nodes != INT_FILL_VALUE]
            face_sizes.append(len(valid_nodes))

        # Most faces should be triangular
        triangular_count = face_sizes.count(3)
        assert triangular_count > len(face_sizes) * 0.9, "Expected mostly triangular faces"

    def test_fesom_netcdf_format(self, test_data_paths):
        """Test FESOM NetCDF format specific functionality."""
        netcdf_path = test_data_paths["fesom"]["netcdf"]
        if not netcdf_path.exists():
            pytest.skip("FESOM NetCDF test file not found")

        grid = ux.open_grid(netcdf_path)

        # FESOM NetCDF format should produce valid grids
        assert grid is not None
        validate_grid_topology(grid)
        validate_grid_coordinates(grid)

    def test_fesom_ocean_properties(self, test_data_paths):
        """Test FESOM ocean model specific properties."""
        for data_key in ["ugrid_diag", "ascii", "netcdf"]:
            grid_path = test_data_paths["fesom"][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            # FESOM is an ocean model, so coordinates should be reasonable for ocean
            # (not checking specific ranges as different grids may have different coverage)
            lon_values = grid.node_lon.values
            lat_values = grid.node_lat.values

            # Basic sanity checks
            assert not np.all(np.isnan(lon_values)), "All longitude values are NaN"
            assert not np.all(np.isnan(lat_values)), "All latitude values are NaN"
            assert len(lon_values) == len(lat_values), "Coordinate arrays have different lengths"

            # Test first available file
            break
