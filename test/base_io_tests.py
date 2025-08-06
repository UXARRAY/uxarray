"""
Base IO test classes for uxarray grid formats.

This module provides reusable test classes that can be inherited by individual
format-specific test modules. It follows pytest best practices with parametrized
tests and fixture-based design.

Usage:
    class TestSpecificFormat(BaseIOTests):
        format_configs = [("format_name", "data_key")]

        def test_format_specific_feature(self):
            # Additional format-specific tests
"""

import pytest
import numpy as np
import xarray as xr
import uxarray as ux
from pathlib import Path
import tempfile
import os
from numpy.testing import assert_array_equal, assert_allclose
from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE, INT_FILL_VALUE

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))





class BaseIOReaderTests:
    """Base class for testing grid reading functionality.

    Subclasses should define:
        format_configs: List of (format_name, data_key) tuples to test
    """

    format_configs = []  # Override in subclasses

    @pytest.mark.io
    def test_read_grid_basic(self, test_data_paths):
        """Test basic grid reading functionality for all configured formats."""
        for format_name, data_key in self.format_configs:
            self._test_read_grid_single(test_data_paths, format_name, data_key)

    def _test_read_grid_single(self, test_data_paths, format_name, data_key):
        """Test basic grid reading functionality for a single format/data combination."""
        grid_path = test_data_paths[format_name][data_key]
        if not grid_path.exists():
            return  # Skip if test file not found

        # Handle special cases for MPAS dual mesh
        if format_name == "mpas":
            grid = ux.open_grid(grid_path, use_dual=False)
        else:
            grid = ux.open_grid(grid_path)

        # Basic validation
        assert grid is not None
        assert hasattr(grid, 'face_node_connectivity')
        assert hasattr(grid, 'node_lon')
        assert hasattr(grid, 'node_lat')

        # Validate grid structure
        self._validate_grid_structure(grid)

    def _validate_grid_structure(self, grid):
        """Common grid structure validation."""
        # Check required dimensions
        required_dims = ['n_node', 'n_face']
        for dim in required_dims:
            assert dim in grid._ds.dims

        # Check connectivity array properties - allow both int32 and int64
        assert grid.face_node_connectivity.dtype in [INT_DTYPE, np.int32, np.int64]
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE

        # Check coordinate consistency
        assert len(grid.node_lon) == len(grid.node_lat)
        assert grid.node_lon.size > 0
        assert grid.node_lat.size > 0

        # Validate grid
        grid.validate()


class BaseIOWriterTests:
    """Base class for testing grid writing functionality.

    Subclasses should define:
        writable_formats: List of format names that support writing
    """

    writable_formats = []  # Override in subclasses

    @pytest.mark.io
    def test_write_format(self, test_data_paths, temp_output_dir):
        """Test writing to all supported formats."""
        for format_name in self.writable_formats:
            self._test_write_single_format(test_data_paths, temp_output_dir, format_name)

    def _test_write_single_format(self, test_data_paths, temp_output_dir, format_name):
        """Test writing to supported formats."""
        # Load a reference grid
        ref_grid = ux.open_grid(test_data_paths["ugrid"]["ne30"])

        output_path = temp_output_dir / f"test_output.{self._get_extension(format_name)}"

        # Convert and write
        dataset = ref_grid.to_xarray(format_name)
        dataset.to_netcdf(output_path)

        # Verify file exists and is readable
        assert output_path.exists()
        reloaded_grid = ux.open_grid(output_path)
        assert reloaded_grid is not None

    def _get_extension(self, format_name):
        """Get appropriate file extension for format."""
        extensions = {
            "UGRID": "nc",
            "Exodus": "exo",
            "ESMF": "nc"
        }
        return extensions.get(format_name, "nc")


class BaseIORoundTripTests:
    """Base class for testing round-trip consistency.

    Subclasses should define:
        round_trip_configs: List of (format_name, data_key, output_format) tuples
    """

    round_trip_configs = []  # Override in subclasses

    @pytest.mark.io
    @pytest.mark.slow
    def test_round_trip_consistency(self, test_data_paths, temp_output_dir):
        """Test round-trip consistency for all configured formats."""
        for format_name, data_key, output_format in self.round_trip_configs:
            self._test_round_trip_single(test_data_paths, temp_output_dir, format_name, data_key, output_format)

    def _test_round_trip_single(self, test_data_paths, temp_output_dir, format_name, data_key, output_format):
        """Test that data survives round-trip conversion for a single configuration."""
        input_path = test_data_paths[format_name][data_key]
        if not input_path.exists():
            return  # Skip if test file not found

        # Load original (handle MPAS special case)
        if format_name == "mpas":
            original_grid = ux.open_grid(input_path, use_dual=False)
        else:
            original_grid = ux.open_grid(input_path)

        # Write in target format
        output_path = temp_output_dir / f"roundtrip.{self._get_extension(output_format)}"
        dataset = original_grid.to_xarray(output_format)
        dataset.to_netcdf(output_path)

        # Read back
        reloaded_grid = ux.open_grid(output_path)

        # Validate consistency
        self._validate_round_trip_consistency(original_grid, reloaded_grid)

    def _validate_round_trip_consistency(self, original, reloaded):
        """Validate that grids are consistent after round-trip."""
        # Topology must be exactly preserved
        assert_array_equal(
            original.face_node_connectivity.values,
            reloaded.face_node_connectivity.values,
            err_msg="Face connectivity mismatch after round-trip"
        )

        # Coordinates should be numerically close
        assert_allclose(
            original.node_lon.values,
            reloaded.node_lon.values,
            rtol=ERROR_TOLERANCE,
            err_msg="Longitude mismatch after round-trip"
        )
        assert_allclose(
            original.node_lat.values,
            reloaded.node_lat.values,
            rtol=ERROR_TOLERANCE,
            err_msg="Latitude mismatch after round-trip"
        )

    def _get_extension(self, format_name):
        """Get appropriate file extension for format."""
        extensions = {
            "UGRID": "nc",
            "Exodus": "exo",
            "ESMF": "nc"
        }
        return extensions.get(format_name, "nc")


class BaseIOEdgeCaseTests:
    """Base class for testing edge cases and error conditions."""

    @pytest.mark.io
    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        with pytest.raises((FileNotFoundError, OSError)):
            ux.open_grid("nonexistent_file.nc")

    @pytest.mark.io
    def test_corrupted_file_handling(self):
        """Test handling of corrupted files - Fixed for Windows compatibility."""
        # Use a different approach that's Windows-compatible
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            temp_path = f.name
            f.write(b"corrupted data")

        try:
            # File is now closed, safe to access on Windows
            with pytest.raises(Exception):  # Should raise some kind of parsing error
                ux.open_grid(temp_path)
        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except (OSError, PermissionError):
                pass  # File might still be in use on Windows

    @pytest.mark.io
    def test_standardized_dtype_and_fill_values(self, test_data_paths):
        """Test that formats use standardized dtype and fill values."""
        # This is a base test - subclasses can override with format-specific files
        pass


class BaseIODatasetTests:
    """Base class for testing dataset (grid + data) functionality."""

    @pytest.mark.io
    def test_dataset_loading_basic(self):
        """Basic dataset loading test - override in subclasses."""
        pass


class BaseIOPerformanceTests:
    """Base class for performance-related IO tests."""

    @pytest.mark.io
    @pytest.mark.performance
    def test_lazy_loading_basic(self, test_data_paths):
        """Basic lazy loading test - override in subclasses with specific formats."""
        pass


# Utility functions that can be used by all format tests
def validate_grid_topology(grid):
    """Validate basic grid topology properties."""
    assert grid.n_node > 0
    assert grid.n_face > 0
    assert grid.face_node_connectivity is not None

    # Check that connectivity indices are valid
    max_node_index = np.nanmax(grid.face_node_connectivity.values)
    assert max_node_index < grid.n_node, "Invalid node indices in connectivity"


def validate_grid_coordinates(grid):
    """Validate grid coordinate properties."""
    # Check longitude range (handle both degrees and radians)
    lon_values = grid.node_lon.values
    lon_max = np.max(np.abs(lon_values))

    if lon_max > 2*np.pi:
        # Likely in degrees
        assert np.all((lon_values >= -360) & (lon_values <= 360)), "Longitude out of valid range for degrees"
    else:
        # Likely in radians
        assert np.all((lon_values >= -2*np.pi) & (lon_values <= 2*np.pi)), "Longitude out of valid range for radians"

    # Check latitude range (handle both degrees and radians)
    lat_values = grid.node_lat.values
    lat_max = np.max(np.abs(lat_values))

    if lat_max > np.pi:
        # Likely in degrees
        assert np.all((lat_values >= -90) & (lat_values <= 90)), "Latitude out of valid range for degrees"
    else:
        # Likely in radians
        assert np.all((lat_values >= -np.pi/2) & (lat_values <= np.pi/2)), "Latitude out of valid range for radians"


def compare_grids_topology(grid1, grid2, rtol=ERROR_TOLERANCE):
    """Compare topology between two grids."""
    # Face connectivity must be identical
    assert_array_equal(
        grid1.face_node_connectivity.values,
        grid2.face_node_connectivity.values,
        err_msg="Face connectivity mismatch"
    )

    # Coordinates should be close
    assert_allclose(
        grid1.node_lon.values,
        grid2.node_lon.values,
        rtol=rtol,
        err_msg="Node longitude mismatch"
    )
    assert_allclose(
        grid1.node_lat.values,
        grid2.node_lat.values,
        rtol=rtol,
        err_msg="Node latitude mismatch"
    )
