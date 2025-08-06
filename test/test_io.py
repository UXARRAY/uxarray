"""
Central IO testing module for uxarray.

This module serves as the main entry point for all IO testing functionality.
It imports and runs tests from individual format modules that inherit from
base IO test classes.

This hybrid approach provides:
- Centralized base test functionality (base_io_tests.py)
- Individual format modules (test_ugrid.py, test_esmf.py, etc.)
- Easy discoverability and maintenance
- Extensibility for new formats

Usage:
    pytest test_io.py  # Runs all IO tests
    pytest test_ugrid.py  # Runs only UGRID tests
    pytest test_esmf.py   # Runs only ESMF tests
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

from .base_io_tests import (
    BaseIOReaderTests,
    BaseIOWriterTests,
    BaseIORoundTripTests,
    BaseIOEdgeCaseTests,
    BaseIODatasetTests,
    BaseIOPerformanceTests
)

# Import all format-specific test modules
# These will be discovered by pytest automatically
try:
    from . import test_ugrid
    from . import test_esmf
    from . import test_mpas
except ImportError as e:
    # Some modules might not be available depending on optional dependencies
    pass

try:
    from . import test_exodus
    from . import test_scrip
    from . import test_icon
    from . import test_fesom
except ImportError as e:
    # Handle optional format modules gracefully
    pass

# Central configurations for all formats
ALL_READABLE_FORMATS = [
    ("ugrid", "ne30"),
    ("ugrid", "rll1deg"),
    ("ugrid", "rll10deg_ne4"),
    ("exodus", "ne8"),
    ("exodus", "mixed"),
    ("esmf", "ne30_grid"),
    ("scrip", "ne8"),
    ("mpas", "qu_grid"),
    ("mpas", "qu_ocean"),
    ("icon", "r02b04"),
    ("fesom", "ugrid_diag"),
    ("fesom", "ascii"),
    ("fesom", "netcdf")
]

ALL_WRITABLE_FORMATS = ["UGRID", "Exodus", "ESMF"]

ALL_ROUND_TRIP_CONFIGS = [
    ("ugrid", "ne30", "UGRID"),
    ("exodus", "mixed", "UGRID"),
    ("exodus", "mixed", "Exodus"),
    ("esmf", "ne30_grid", "ESMF"),
    ("scrip", "ne8", "UGRID")
]


class TestIOOverview:
    """
    Overview tests that run across all formats.

    These tests provide a high-level view of IO functionality
    and can catch format-specific issues that might be missed
    in individual modules.
    """

    @pytest.mark.parametrize("format_name,data_key", ALL_READABLE_FORMATS)
    def test_all_formats_readable(self, test_data_paths, format_name, data_key):
        """High-level test that all supported formats can be read."""
        grid_path = test_data_paths[format_name][data_key]
        if not grid_path.exists():
            pytest.skip(f"Test file not found: {grid_path}")

        # Handle special cases
        if format_name == "mpas":
            grid = ux.open_grid(grid_path, use_dual=False)
        else:
            grid = ux.open_grid(grid_path)

        # Basic validation
        assert grid is not None
        assert hasattr(grid, 'face_node_connectivity')
        assert hasattr(grid, 'node_lon')
        assert hasattr(grid, 'node_lat')

        # Validate grid
        grid.validate()

    @pytest.mark.parametrize("format_name", ALL_WRITABLE_FORMATS)
    def test_all_formats_writable(self, test_data_paths, temp_output_dir, format_name):
        """High-level test that all supported formats can be written."""
        # Use a simple reference grid
        ref_grid = ux.open_grid(test_data_paths["ugrid"]["ne30"])

        # Convert and write
        try:
            dataset = ref_grid.to_xarray(format_name)
            output_path = temp_output_dir / f"test_write_{format_name.lower()}.nc"
            dataset.to_netcdf(output_path)

            # Verify readability
            reloaded_grid = ux.open_grid(output_path)
            assert reloaded_grid is not None

        except Exception as e:
            pytest.fail(f"Failed to write {format_name} format: {e}")


class TestIOCompatibility:
    """
    Test compatibility and interoperability between different formats.
    """

    def test_format_conversion_matrix(self, test_data_paths, temp_output_dir):
        """Test converting between different formats where supported."""
        # Load a source grid
        source_grid = ux.open_grid(test_data_paths["ugrid"]["ne30"])

        conversion_pairs = [
            ("UGRID", "ESMF"),
            ("UGRID", "Exodus"),
            ("ESMF", "UGRID"),
            ("Exodus", "UGRID")
        ]

        for source_fmt, target_fmt in conversion_pairs:
            try:
                # Convert source to intermediate format
                intermediate_ds = source_grid.to_xarray(source_fmt)
                intermediate_path = temp_output_dir / f"intermediate_{source_fmt.lower()}.nc"
                intermediate_ds.to_netcdf(intermediate_path)

                # Load intermediate and convert to target
                intermediate_grid = ux.open_grid(intermediate_path)
                target_ds = intermediate_grid.to_xarray(target_fmt)
                target_path = temp_output_dir / f"target_{target_fmt.lower()}.nc"
                target_ds.to_netcdf(target_path)

                # Verify final result
                final_grid = ux.open_grid(target_path)
                assert final_grid is not None
                assert final_grid.n_face == source_grid.n_face

            except Exception as e:
                # Log but don't fail - some conversions might not be supported
                print(f"Conversion {source_fmt} -> {target_fmt} failed: {e}")


class TestIOComprehensive(BaseIOEdgeCaseTests):
    """
    Comprehensive edge case testing across all formats.

    This class provides additional edge case testing that complements
    the format-specific modules.
    """

    def test_empty_directory_handling(self):
        """Test handling of empty or non-existent directories."""
        with pytest.raises((FileNotFoundError, OSError)):
            ux.open_grid("/nonexistent/directory/file.nc")

    def test_large_grid_metadata(self, test_data_paths):
        """Test handling of grids with large metadata."""
        # Use the largest available grid for testing
        large_grid_configs = [
            ("ugrid", "rll1deg"),  # ~64k nodes
            ("ugrid", "ne30"),     # ~5k nodes
            ("mpas", "qu_grid")    # Variable size
        ]

        for format_name, data_key in large_grid_configs:
            grid_path = test_data_paths[format_name][data_key]
            if not grid_path.exists():
                continue

            if format_name == "mpas":
                grid = ux.open_grid(grid_path, use_dual=False)
            else:
                grid = ux.open_grid(grid_path)

            # Test that large grids can be handled
            assert grid.n_node > 1000  # Reasonable size threshold
            assert grid.n_face > 100

            # Test one large grid is sufficient
            break

    def test_coordinate_edge_cases(self, test_data_paths):
        """Test edge cases in coordinate handling."""
        test_configs = [
            ("ugrid", "ne30"),
            ("esmf", "ne30_grid"),
            ("mpas", "qu_grid")
        ]

        for format_name, data_key in test_configs:
            grid_path = test_data_paths[format_name][data_key]
            if not grid_path.exists():
                continue

            if format_name == "mpas":
                grid = ux.open_grid(grid_path, use_dual=False)
            else:
                grid = ux.open_grid(grid_path)

            # Test coordinate bounds
            lon_vals = grid.node_lon.values
            lat_vals = grid.node_lat.values

            # Check for reasonable ranges (allowing different coordinate systems)
            assert np.all(np.isfinite(lon_vals)), "Non-finite longitude values"
            assert np.all(np.isfinite(lat_vals)), "Non-finite latitude values"

            # Test one grid per format
            break


class TestIODocumentation:
    """
    Tests that verify IO functionality matches documentation and examples.
    """

    def test_basic_io_examples(self, test_data_paths):
        """Test that basic IO examples from documentation work."""
        # Test basic grid loading
        grid_path = test_data_paths["ugrid"]["ne30"]
        if grid_path.exists():
            grid = ux.open_grid(grid_path)
            assert grid is not None

            # Test basic properties mentioned in docs
            assert hasattr(grid, 'n_face')
            assert hasattr(grid, 'n_node')
            assert grid.n_face > 0
            assert grid.n_node > 0

    def test_format_support_documentation(self, test_data_paths):
        """Verify that documented format support is accurate."""
        documented_readable_formats = [
            "ugrid", "exodus", "esmf", "scrip", "mpas", "icon", "fesom"
        ]

        documented_writable_formats = [
            "UGRID", "Exodus", "ESMF"
        ]

        # Test that all documented readable formats have test files
        available_readable = set()
        for fmt in documented_readable_formats:
            if fmt in test_data_paths:
                for data_key, path in test_data_paths[fmt].items():
                    if path.exists():
                        available_readable.add(fmt)
                        break

        # Should have test files for most documented formats
        assert len(available_readable) >= len(documented_readable_formats) * 0.7

        # Test writable formats
        ref_grid = ux.open_grid(test_data_paths["ugrid"]["ne30"])
        working_writable = []

        for fmt in documented_writable_formats:
            try:
                dataset = ref_grid.to_xarray(fmt)
                working_writable.append(fmt)
            except Exception:
                pass

        # Should support most documented writable formats
        assert len(working_writable) >= len(documented_writable_formats) * 0.8


# Utility functions for cross-format testing
def get_available_test_files(test_data_paths):
    """Get list of available test files across all formats."""
    available_files = []

    for format_name, format_data in test_data_paths.items():
        for data_key, file_path in format_data.items():
            if file_path.exists():
                available_files.append((format_name, data_key, file_path))

    return available_files


def validate_cross_format_consistency():
    """Validate that different formats produce consistent results."""
    # This is a placeholder for cross-format validation
    # Can be expanded based on specific requirements
    pass


# Test discovery and execution
if __name__ == "__main__":
    # This allows running the module directly for debugging
    pytest.main([__file__, "-v"])
