"""
Exodus format tests using base IO test classes.

This module tests Exodus-specific functionality while inheriting common
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

# Exodus-specific test configurations
EXODUS_READ_CONFIGS = [
    ("exodus", "ne8"),
    ("exodus", "mixed")
]

EXODUS_WRITE_FORMATS = ["Exodus"]

EXODUS_ROUND_TRIP_CONFIGS = [
    ("exodus", "ne8", "Exodus"),
    ("exodus", "mixed", "Exodus")
]


class TestExodusReader(BaseIOReaderTests):
    """Test Exodus reading functionality."""

    format_configs = EXODUS_READ_CONFIGS

    def test_exodus_basic_structure(self, test_data_paths):
        """Test basic Exodus file structure."""
        grid_path = test_data_paths["exodus"]["ne8"]
        if not grid_path.exists():
            pytest.skip("Exodus ne8 test file not found")

        grid = ux.open_grid(grid_path)

        # Basic validation
        assert grid is not None
        validate_grid_topology(grid)
        validate_grid_coordinates(grid)

    def test_exodus_mixed_face_types(self, test_data_paths):
        """Test Exodus file with mixed face types (triangles and quadrilaterals)."""
        grid_path = test_data_paths["exodus"]["mixed"]
        if not grid_path.exists():
            pytest.skip("Exodus mixed test file not found")

        grid = ux.open_grid(grid_path)

        # Basic validation
        assert grid is not None
        validate_grid_topology(grid)
        validate_grid_coordinates(grid)

        # Check that we have mixed face types by looking at connectivity
        connectivity = grid.face_node_connectivity.values
        face_sizes = []
        for face_idx in range(grid.n_face):
            face_nodes = connectivity[face_idx]
            valid_nodes = face_nodes[face_nodes != INT_FILL_VALUE]
            face_sizes.append(len(valid_nodes))

        # Should have both triangles (3 nodes) and quadrilaterals (4 nodes)
        unique_sizes = set(face_sizes)
        assert len(unique_sizes) > 1, "Expected mixed face types but found uniform faces"


class TestExodusWriter(BaseIOWriterTests):
    """Test Exodus writing functionality."""

    writable_formats = EXODUS_WRITE_FORMATS

    def test_exodus_write_basic(self, test_data_paths, temp_output_dir):
        """Test basic Exodus writing functionality."""
        # Load a reference grid
        ref_grid_path = test_data_paths["exodus"]["ne8"]
        if not ref_grid_path.exists():
            pytest.skip("Exodus ne8 test file not found")

        ref_grid = ux.open_grid(ref_grid_path)
        output_path = temp_output_dir / "test_output.exo"

        # Convert to Exodus format and write
        exodus_dataset = ref_grid.to_xarray("Exodus")
        exodus_dataset.to_netcdf(output_path)

        # Verify file exists and is readable
        assert output_path.exists()
        reloaded_grid = ux.open_grid(output_path)
        assert reloaded_grid is not None

        # Basic structure validation
        validate_grid_topology(reloaded_grid)


class TestExodusRoundTrip(BaseIORoundTripTests):
    """Test Exodus round-trip consistency."""

    round_trip_configs = EXODUS_ROUND_TRIP_CONFIGS

    def test_exodus_mixed_face_round_trip(self, test_data_paths, temp_output_dir):
        """Test round-trip consistency for mixed face types."""
        grid_path = test_data_paths["exodus"]["mixed"]
        if not grid_path.exists():
            pytest.skip("Exodus mixed test file not found")

        # Load original
        original_grid = ux.open_grid(grid_path)

        # Test round-trip through UGRID format
        ugrid_path = temp_output_dir / "test_ugrid.nc"
        ugrid_dataset = original_grid.to_xarray("UGRID")
        ugrid_dataset.to_netcdf(ugrid_path)

        # Test round-trip through Exodus format
        exodus_path = temp_output_dir / "test_exodus.exo"
        exodus_dataset = original_grid.to_xarray("Exodus")
        exodus_dataset.to_netcdf(exodus_path)

        # Load both back
        ugrid_reloaded = ux.open_grid(ugrid_path)
        exodus_reloaded = ux.open_grid(exodus_path)

        # Validate consistency
        self._validate_round_trip_consistency(original_grid, ugrid_reloaded)
        self._validate_round_trip_consistency(original_grid, exodus_reloaded)

        # Clean up
        ugrid_reloaded._ds.close()
        exodus_reloaded._ds.close()


class TestExodusEdgeCases(BaseIOEdgeCaseTests):
    """Test Exodus edge cases and error conditions."""

    def test_standardized_dtype_and_fill_values(self, test_data_paths):
        """Test that Exodus files use standardized dtype and fill values."""
        test_files = [
            ("exodus", "ne8"),
            ("exodus", "mixed")
        ]

        for format_name, data_key in test_files:
            grid_path = test_data_paths[format_name][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            # Check dtype and fill value
            assert grid.face_node_connectivity.dtype in [INT_DTYPE, np.int32, np.int64]
            assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE

    def test_exodus_init_from_vertices(self):
        """Test creating Exodus grid from vertices."""
        # Create a simple single face grid from vertices
        verts = [[[0, 0], [2, 0], [0, 2], [2, 2]]]
        grid = ux.open_grid(verts)

        assert grid is not None
        assert grid.n_face == 1
        assert grid.n_node == 4

        validate_grid_topology(grid)


class TestExodusDatasets(BaseIODatasetTests):
    """Test Exodus dataset functionality."""

    def test_exodus_dataset_basic(self, test_data_paths):
        """Basic Exodus dataset validation."""
        for data_key in ["ne8", "mixed"]:
            grid_path = test_data_paths["exodus"][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            assert grid is not None
            validate_grid_topology(grid)
            validate_grid_coordinates(grid)

            # Test first file is sufficient
            break


class TestExodusPerformance(BaseIOPerformanceTests):
    """Test Exodus performance characteristics."""

    @pytest.mark.parametrize("data_key", ["ne8", "mixed"])
    def test_lazy_loading(self, test_data_paths, data_key):
        """Test that Exodus grid loading is reasonably fast."""
        import time

        grid_path = test_data_paths["exodus"][data_key]
        if not grid_path.exists():
            pytest.skip(f"Exodus test file not found: {grid_path}")

        start_time = time.time()
        grid = ux.open_grid(grid_path)
        load_time = time.time() - start_time

        # Basic validation that grid loaded successfully
        assert grid is not None
        assert hasattr(grid, 'node_lon')

        # Loading should complete in reasonable time
        assert load_time < 30.0, f"Exodus loading took {load_time:.2f}s, which seems excessive"


class TestExodusSpecialCases:
    """Test Exodus-specific special cases and functionality."""

    def test_exodus_face_type_handling(self, test_data_paths):
        """Test handling of different face types in Exodus format."""
        grid_path = test_data_paths["exodus"]["mixed"]
        if not grid_path.exists():
            pytest.skip("Exodus mixed test file not found")

        grid = ux.open_grid(grid_path)

        # Analyze face types
        connectivity = grid.face_node_connectivity.values
        face_types = []

        for face_idx in range(grid.n_face):
            face_nodes = connectivity[face_idx]
            valid_nodes = face_nodes[face_nodes != INT_FILL_VALUE]
            face_types.append(len(valid_nodes))

        # Should have mixed face types
        unique_face_types = set(face_types)
        assert len(unique_face_types) > 1, "Expected mixed face types"

        # Common face types in Exodus are triangles (3) and quads (4)
        for face_type in unique_face_types:
            assert face_type >= 3, f"Invalid face type with {face_type} nodes"
            assert face_type <= 8, f"Unexpectedly large face with {face_type} nodes"

    def test_exodus_coordinate_system(self, test_data_paths):
        """Test Exodus coordinate system handling."""
        for data_key in ["ne8", "mixed"]:
            grid_path = test_data_paths["exodus"][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            # Validate coordinate properties
            validate_grid_coordinates(grid)

            # Test one file is sufficient
            break

    def test_exodus_format_conversion_consistency(self, test_data_paths, temp_output_dir):
        """Test that Exodus format conversions maintain data consistency."""
        grid_path = test_data_paths["exodus"]["ne8"]
        if not grid_path.exists():
            pytest.skip("Exodus ne8 test file not found")

        original_grid = ux.open_grid(grid_path)

        # Convert to different formats and back
        formats_to_test = ["UGRID", "Exodus"]

        for target_format in formats_to_test:
            # Convert to target format
            converted_dataset = original_grid.to_xarray(target_format)

            # Write and reload
            if target_format == "UGRID":
                output_path = temp_output_dir / f"converted_{target_format.lower()}.nc"
            else:
                output_path = temp_output_dir / f"converted_{target_format.lower()}.exo"

            converted_dataset.to_netcdf(output_path)
            reloaded_grid = ux.open_grid(output_path)

            # Validate topology preservation
            assert_array_equal(
                original_grid.face_node_connectivity.values,
                reloaded_grid.face_node_connectivity.values,
                err_msg=f"Face connectivity mismatch after {target_format} conversion"
            )

            # Validate coordinate preservation
            assert_allclose(
                original_grid.node_lon.values,
                reloaded_grid.node_lon.values,
                rtol=1e-10,
                err_msg=f"Node longitude mismatch after {target_format} conversion"
            )

            reloaded_grid._ds.close()
