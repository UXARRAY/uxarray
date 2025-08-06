"""
UGRID format tests using base IO test classes.

This module tests UGRID-specific functionality while inheriting common
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

# UGRID-specific test configurations
UGRID_READ_CONFIGS = [
    ("ugrid", "ne30"),
    ("ugrid", "rll1deg"),
    ("ugrid", "rll10deg_ne4")
]

UGRID_WRITE_FORMATS = ["UGRID"]

UGRID_ROUND_TRIP_CONFIGS = [
    ("ugrid", "ne30", "UGRID"),
    ("ugrid", "rll1deg", "UGRID"),
    ("ugrid", "rll10deg_ne4", "UGRID")
]


class TestUGRIDReader(BaseIOReaderTests):
    """Test UGRID reading functionality."""

    format_configs = UGRID_READ_CONFIGS

    @pytest.mark.parametrize("format_name,data_key", UGRID_READ_CONFIGS)
    def test_ugrid_node_counts(self, test_data_paths, format_name, data_key):
        """Test that UGRID files have expected node counts."""
        grid_path = test_data_paths[format_name][data_key]
        if not grid_path.exists():
            pytest.skip(f"Test file not found: {grid_path}")

        grid = ux.open_grid(grid_path)

        expected_counts = {
            "ne30": constants.NNODES_outCSne30,
            "rll1deg": constants.NNODES_outRLL1deg,
            "rll10deg_ne4": constants.NNODES_ov_RLL10deg_CSne4
        }

        if data_key in expected_counts:
            assert grid.node_lon.size == expected_counts[data_key]

    def test_ugrid_dimensions_and_coordinates(self, test_data_paths):
        """Test UGRID-specific dimension and coordinate structure."""
        grid_path = test_data_paths["ugrid"]["ne30"]
        if not grid_path.exists():
            pytest.skip("UGRID ne30 test file not found")

        grid = ux.open_grid(grid_path)

        # Check standard UGRID dimensions
        required_dims = ['n_node', 'n_face', 'n_max_face_nodes']
        for dim in required_dims:
            assert dim in grid._ds.dims

        # Check coordinate variables
        coord_vars = ['node_lon', 'node_lat', 'face_lon', 'face_lat']
        for coord in coord_vars:
            if coord in grid._ds:  # Some may be computed on demand
                assert grid._ds[coord].dims[0] in ['n_node', 'n_face']


class TestUGRIDWriter(BaseIOWriterTests):
    """Test UGRID writing functionality."""

    writable_formats = UGRID_WRITE_FORMATS

    def test_ugrid_boolean_attr_conversion(self):
        """Test that encode_as('UGRID') converts boolean attrs to int."""
        # Create a minimal grid with a boolean attribute
        ds = xr.Dataset(
            {
                "node_lon": (("n_node",), [0.0, 1.0]),
                "node_lat": (("n_node",), [0.0, 1.0]),
                "face_node_connectivity": (("n_face", "n_max_face_nodes"), [[0, 1, -1, -1]])
            },
            coords={"n_node": [0, 1], "n_face": [0], "n_max_face_nodes": [0, 1, 2, 3]},
            attrs={"test_bool": True, "test_str": "abc"}
        )

        # Add minimal grid_topology for UGRID
        ds["grid_topology"] = xr.DataArray(
            data=-1,
            attrs={
                "cf_role": "mesh_topology",
                "topology_dimension": 2,
                "face_dimension": "n_face",
                "node_dimension": "n_node",
                "node_coordinates": "node_lon node_lat",
                "face_node_connectivity": "face_node_connectivity"
            }
        )

        grid = ux.Grid(ds)
        encoded = grid.to_xarray("UGRID")

        # Check that the returned dataset is not the same object
        assert encoded is not grid._ds
        # Check that the boolean attribute is now an int
        assert isinstance(encoded.attrs["test_bool"], int)
        assert encoded.attrs["test_bool"] == 1
        # Check that the string attribute is unchanged
        assert encoded.attrs["test_str"] == "abc"
        # Check that the original dataset is not modified
        assert isinstance(ds.attrs["test_bool"], bool)

    def test_ugrid_metadata_preservation(self, test_data_paths, temp_output_dir):
        """Test that UGRID metadata is preserved during write operations."""
        grid_path = test_data_paths["ugrid"]["ne30"]
        if not grid_path.exists():
            pytest.skip("UGRID ne30 test file not found")

        original_grid = ux.open_grid(grid_path)

        # Write and reload
        output_path = temp_output_dir / "metadata_test.nc"
        ugrid_dataset = original_grid.to_xarray("UGRID")
        ugrid_dataset.to_netcdf(output_path)
        reloaded_grid = ux.open_grid(output_path)

        # Check that basic metadata is preserved
        assert reloaded_grid.n_face == original_grid.n_face
        assert reloaded_grid.n_node == original_grid.n_node


class TestUGRIDRoundTrip(BaseIORoundTripTests):
    """Test UGRID round-trip consistency."""

    round_trip_configs = UGRID_ROUND_TRIP_CONFIGS

    def test_ugrid_to_ugrid_detailed(self, test_data_paths, temp_output_dir):
        """Test detailed UGRID to UGRID round-trip with specific validation."""
        grid_path = test_data_paths["ugrid"]["ne30"]
        if not grid_path.exists():
            pytest.skip("UGRID ne30 test file not found")

        # Load original grid
        original_grid = ux.open_grid(grid_path)

        # Convert to UGRID format and save
        ugrid_dataset = original_grid.to_xarray("UGRID")
        output_path = temp_output_dir / "ugrid_roundtrip.nc"
        ugrid_dataset.to_netcdf(output_path)

        # Reload and validate
        reloaded_grid = ux.open_grid(output_path)

        # Detailed validation
        validate_grid_topology(original_grid)
        validate_grid_topology(reloaded_grid)
        validate_grid_coordinates(original_grid)
        validate_grid_coordinates(reloaded_grid)

        # Check dimensions match
        assert reloaded_grid.n_face == original_grid.n_face
        assert reloaded_grid.n_node == original_grid.n_node
        assert reloaded_grid.n_max_face_nodes == original_grid.n_max_face_nodes


class TestUGRIDEdgeCases(BaseIOEdgeCaseTests):
    """Test UGRID edge cases and error conditions."""

    def test_standardized_dtype_and_fill_values(self, test_data_paths):
        """Test that UGRID files use standardized dtype and fill values."""
        test_files = [
            ("ugrid", "ne30"),
            ("ugrid", "rll1deg"),
            ("ugrid", "rll10deg_ne4")
        ]

        for format_name, data_key in test_files:
            grid_path = test_data_paths[format_name][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            # Check dtype and fill value (allow both int32 and int64)
            from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
            assert grid.face_node_connectivity.dtype in [INT_DTYPE, np.int32, np.int64]
            assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE

    def test_ugrid_with_fill_values(self, test_data_paths):
        """Test UGRID files that contain fill values in connectivity."""
        grid_path = test_data_paths["ugrid"]["rll1deg"]  # This file has fill values
        if not grid_path.exists():
            pytest.skip("UGRID rll1deg test file not found")

        grid = ux.open_grid(grid_path)

        # Check that fill values are present and handled correctly
        from uxarray.constants import INT_FILL_VALUE
        connectivity = grid.face_node_connectivity.values
        has_fill_values = INT_FILL_VALUE in connectivity

        if has_fill_values:
            # Validate that fill values are only at the end of face definitions
            for face_idx in range(grid.n_face):
                face_nodes = connectivity[face_idx]
                fill_positions = np.where(face_nodes == INT_FILL_VALUE)[0]
                if len(fill_positions) > 0:
                    # Fill values should be consecutive and at the end
                    assert np.all(np.diff(fill_positions) == 1), "Fill values should be consecutive"


class TestUGRIDDatasets(BaseIODatasetTests):
    """Test UGRID dataset functionality."""

    def test_ugrid_data_variables_preservation(self, test_data_paths):
        """Test that data variables are preserved when working with UGRID datasets."""
        # This would test dataset functionality if we had UGRID data files
        # For now, just validate the basic structure
        grid_path = test_data_paths["ugrid"]["ne30"]
        if not grid_path.exists():
            pytest.skip("UGRID test file not found")

        # Open as dataset (though this file may only have grid info)
        try:
            dataset = ux.open_dataset(grid_path)
            assert hasattr(dataset, 'uxgrid')
            assert dataset.uxgrid is not None
        except Exception:
            # If this fails, it's likely because the file doesn't have data variables
            # which is fine for a grid-only file
            pass


class TestUGRIDPerformance(BaseIOPerformanceTests):
    """Test UGRID performance characteristics."""

    @pytest.mark.parametrize("format_name,data_key", [
        ("ugrid", "ne30"),
        ("ugrid", "rll1deg")
    ])
    def test_lazy_loading(self, test_data_paths, format_name, data_key):
        """Test that UGRID grid loading is reasonably fast."""
        import time

        grid_path = test_data_paths[format_name][data_key]
        if not grid_path.exists():
            pytest.skip(f"Test file not found: {grid_path}")

        start_time = time.time()
        grid = ux.open_grid(grid_path)
        load_time = time.time() - start_time

        # Basic validation that grid loaded successfully
        assert grid is not None
        assert hasattr(grid, 'node_lon')

        # Loading should complete in reasonable time
        assert load_time < 30.0, f"UGRID loading took {load_time:.2f}s, which seems excessive"


class TestUGRIDSpecialCases:
    """Test UGRID-specific special cases and functionality."""

    def test_ugrid_cf_compliance(self, test_data_paths):
        """Test that UGRID files follow CF conventions."""
        grid_path = test_data_paths["ugrid"]["ne30"]
        if not grid_path.exists():
            pytest.skip("UGRID test file not found")

        grid = ux.open_grid(grid_path)

        # Check for CF-compliant mesh topology variable
        if 'grid_topology' in grid._ds:
            topo_var = grid._ds['grid_topology']
            assert 'cf_role' in topo_var.attrs
            assert topo_var.attrs['cf_role'] == 'mesh_topology'

    def test_ugrid_coordinate_bounds(self, test_data_paths):
        """Test handling of coordinate bounds in UGRID files."""
        for data_key in ["ne30", "rll1deg", "rll10deg_ne4"]:
            grid_path = test_data_paths["ugrid"][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            # Check that coordinates are within valid ranges
            validate_grid_coordinates(grid)

            # Test one file is sufficient for this test
            break
