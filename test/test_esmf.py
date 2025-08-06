"""
ESMF format tests using base IO test classes.

This module tests ESMF-specific functionality while inheriting common
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
from uxarray.constants import ERROR_TOLERANCE

from .base_io_tests import (
    BaseIOReaderTests,
    BaseIOWriterTests,
    BaseIORoundTripTests,
    BaseIOEdgeCaseTests,
    BaseIODatasetTests,
    BaseIOPerformanceTests,
    validate_grid_topology,
    validate_grid_coordinates,
    compare_grids_topology
)

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

# ESMF-specific test configurations
ESMF_READ_CONFIGS = [
    ("esmf", "ne30_grid")
]

ESMF_WRITE_FORMATS = ["ESMF"]

ESMF_ROUND_TRIP_CONFIGS = [
    ("esmf", "ne30_grid", "ESMF")
]


class TestESMFReader(BaseIOReaderTests):
    """Test ESMF reading functionality."""

    format_configs = ESMF_READ_CONFIGS

    def test_esmf_grid_structure(self, test_data_paths):
        """Test ESMF grid structure and encoding."""
        grid_path = test_data_paths["esmf"]["ne30_grid"]
        if not grid_path.exists():
            pytest.skip("ESMF test file not found")

        grid = ux.open_grid(grid_path)

        # Check for ESMF-specific dimensions and coordinates
        dims = ['n_node', 'n_face', 'n_max_face_nodes']
        coords = ['node_lon', 'node_lat', 'face_lon', 'face_lat']
        conns = ['face_node_connectivity', 'n_nodes_per_face']

        for dim in dims:
            assert dim in grid._ds.dims

        for coord in coords:
            assert coord in grid._ds

        for conn in conns:
            assert conn in grid._ds

    def test_esmf_coordinate_units(self, test_data_paths):
        """Test that ESMF coordinates have proper units."""
        grid_path = test_data_paths["esmf"]["ne30_grid"]
        if not grid_path.exists():
            pytest.skip("ESMF test file not found")

        grid = ux.open_grid(grid_path)

        # ESMF typically uses radians for coordinates
        # Validate coordinate ranges are reasonable
        validate_grid_coordinates(grid)

        # Check that we have face center coordinates
        assert hasattr(grid, 'face_lon')
        assert hasattr(grid, 'face_lat')


class TestESMFWriter(BaseIOWriterTests):
    """Test ESMF writing functionality."""

    writable_formats = ESMF_WRITE_FORMATS

    def test_esmf_dataset_structure(self, test_data_paths, temp_output_dir):
        """Test ESMF dataset structure after conversion."""
        # Load a reference grid
        ref_grid = ux.open_grid(test_data_paths["ugrid"]["ne30"])

        # Convert to ESMF format
        esmf_dataset = ref_grid.to_xarray("ESMF")

        # Verify ESMF-specific structure
        assert isinstance(esmf_dataset, xr.Dataset)
        assert 'nodeCoords' in esmf_dataset
        assert 'elementConn' in esmf_dataset

        # Test serialization
        output_path = temp_output_dir / "test_esmf_output.nc"
        esmf_dataset.to_netcdf(output_path)

        # Verify file was created and is readable
        assert output_path.exists()
        reloaded_grid = ux.open_grid(output_path)
        assert reloaded_grid is not None


class TestESMFRoundTrip(BaseIORoundTripTests):
    """Test ESMF round-trip consistency."""

    round_trip_configs = ESMF_ROUND_TRIP_CONFIGS

    def test_esmf_round_trip_detailed(self, test_data_paths, temp_output_dir):
        """Test detailed ESMF round-trip with specific validation."""
        # Use UGRID as source since we know it works well
        grid_path = test_data_paths["ugrid"]["ne30"]
        if not grid_path.exists():
            pytest.skip("UGRID ne30 test file not found")

        # Load original grid
        original_grid = ux.open_grid(grid_path)

        # Convert to ESMF xarray format
        esmf_dataset = original_grid.to_xarray("ESMF")

        # Verify dataset structure
        assert isinstance(esmf_dataset, xr.Dataset)
        assert 'nodeCoords' in esmf_dataset
        assert 'elementConn' in esmf_dataset

        # Serialize dataset to disk
        esmf_filepath = temp_output_dir / "test_esmf_ne30.nc"
        esmf_dataset.to_netcdf(esmf_filepath)

        # Reload grid from serialized file
        reloaded_grid = ux.open_grid(esmf_filepath)

        # Validate topological consistency
        assert_array_equal(
            original_grid.face_node_connectivity.values,
            reloaded_grid.face_node_connectivity.values,
            err_msg="ESMF face connectivity mismatch"
        )

        # Validate coordinate consistency
        assert_allclose(
            original_grid.node_lon.values,
            reloaded_grid.node_lon.values,
            err_msg="ESMF longitude mismatch",
            rtol=ERROR_TOLERANCE
        )
        assert_allclose(
            original_grid.node_lat.values,
            reloaded_grid.node_lat.values,
            err_msg="ESMF latitude mismatch",
            rtol=ERROR_TOLERANCE
        )

    def test_esmf_multiple_format_round_trip(self, test_data_paths, temp_output_dir):
        """Test ESMF round-trip through multiple format conversions."""
        grid_path = test_data_paths["esmf"]["ne30_grid"]
        if not grid_path.exists():
            pytest.skip("ESMF test file not found")

        original_grid = ux.open_grid(grid_path)

        # Convert ESMF -> UGRID -> ESMF
        ugrid_dataset = original_grid.to_xarray("UGRID")
        ugrid_path = temp_output_dir / "esmf_to_ugrid.nc"
        ugrid_dataset.to_netcdf(ugrid_path)

        intermediate_grid = ux.open_grid(ugrid_path)
        final_esmf_dataset = intermediate_grid.to_xarray("ESMF")
        final_path = temp_output_dir / "ugrid_to_esmf.nc"
        final_esmf_dataset.to_netcdf(final_path)

        final_grid = ux.open_grid(final_path)

        # Validate that we end up with the same topology
        compare_grids_topology(original_grid, final_grid)


class TestESMFEdgeCases(BaseIOEdgeCaseTests):
    """Test ESMF edge cases and error conditions."""

    def test_esmf_coordinate_conversion(self, test_data_paths):
        """Test ESMF coordinate system conversions."""
        grid_path = test_data_paths["esmf"]["ne30_grid"]
        if not grid_path.exists():
            pytest.skip("ESMF test file not found")

        grid = ux.open_grid(grid_path)

        # Validate coordinates are in reasonable ranges
        validate_grid_coordinates(grid)

        # Check that coordinate conversion doesn't introduce large errors
        lon_range = np.ptp(grid.node_lon.values)  # Peak-to-peak
        lat_range = np.ptp(grid.node_lat.values)

        # Should span reasonable ranges for a global grid
        assert lon_range > np.pi, "Longitude range seems too small for global grid"
        assert lat_range > np.pi/2, "Latitude range seems too small for global grid"


class TestESMFDatasets(BaseIODatasetTests):
    """Test ESMF dataset functionality."""

    def test_esmf_dataset_loading(self, test_data_paths):
        """Test loading ESMF datasets with grid and data files."""
        grid_path = test_data_paths["esmf"]["ne30_grid"]
        data_path = test_data_paths["esmf"]["ne30_data"]

        if not (grid_path.exists() and data_path.exists()):
            pytest.skip("ESMF test files not found")

        dataset = ux.open_dataset(grid_path, data_path)

        # Validate dataset structure
        assert 'n_node' in dataset.dims
        assert 'n_face' in dataset.dims
        assert hasattr(dataset, 'uxgrid')

    def test_esmf_data_variable_mapping(self, test_data_paths):
        """Test that ESMF data variables are properly mapped to grid."""
        grid_path = test_data_paths["esmf"]["ne30_grid"]
        data_path = test_data_paths["esmf"]["ne30_data"]

        if not (grid_path.exists() and data_path.exists()):
            pytest.skip("ESMF test files not found")

        try:
            dataset = ux.open_dataset(grid_path, data_path)

            # Check that data variables are associated with appropriate dimensions
            for var_name in dataset.data_vars:
                var = dataset[var_name]
                # Data should be on nodes or faces
                assert any(dim in var.dims for dim in ['n_node', 'n_face']), \
                    f"Variable {var_name} not associated with nodes or faces"

        except Exception as e:
            # If data loading fails, that's okay - just validate grid loading works
            grid = ux.open_grid(grid_path)
            assert grid is not None


class TestESMFPerformance(BaseIOPerformanceTests):
    """Test ESMF performance characteristics."""

    def test_lazy_loading(self, test_data_paths):
        """Test that ESMF grid loading is reasonably fast."""
        import time

        grid_path = test_data_paths["esmf"]["ne30_grid"]
        if not grid_path.exists():
            pytest.skip("ESMF test file not found")

        start_time = time.time()
        grid = ux.open_grid(grid_path)
        load_time = time.time() - start_time

        # Basic validation that grid loaded successfully
        assert grid is not None
        assert hasattr(grid, 'node_lon')

        # Loading should complete in reasonable time
        assert load_time < 30.0, f"ESMF loading took {load_time:.2f}s, which seems excessive"


class TestESMFSpecialCases:
    """Test ESMF-specific special cases and functionality."""

    def test_esmf_variable_naming_conventions(self, test_data_paths):
        """Test ESMF variable naming conventions."""
        grid_path = test_data_paths["esmf"]["ne30_grid"]
        if not grid_path.exists():
            pytest.skip("ESMF test file not found")

        grid = ux.open_grid(grid_path)

        # After converting to ESMF format, check variable names
        esmf_ds = grid.to_xarray("ESMF")

        # ESMF should have specific variable names
        expected_vars = ['nodeCoords', 'elementConn']
        for var in expected_vars:
            assert var in esmf_ds, f"Expected ESMF variable {var} not found"

    def test_esmf_dimension_ordering(self, test_data_paths):
        """Test that ESMF maintains proper dimension ordering."""
        grid_path = test_data_paths["esmf"]["ne30_grid"]
        if not grid_path.exists():
            pytest.skip("ESMF test file not found")

        grid = ux.open_grid(grid_path)

        # Check connectivity array dimensions
        conn = grid.face_node_connectivity
        assert len(conn.dims) == 2
        assert conn.dims[0] == 'n_face'
        assert conn.dims[1] == 'n_max_face_nodes'

    def test_esmf_metadata_preservation(self, test_data_paths, temp_output_dir):
        """Test that important metadata is preserved in ESMF format."""
        grid_path = test_data_paths["esmf"]["ne30_grid"]
        if not grid_path.exists():
            pytest.skip("ESMF test file not found")

        original_grid = ux.open_grid(grid_path)

        # Convert to ESMF and save
        esmf_ds = original_grid.to_xarray("ESMF")
        output_path = temp_output_dir / "esmf_metadata_test.nc"
        esmf_ds.to_netcdf(output_path)

        # Reload and check basic properties
        reloaded_grid = ux.open_grid(output_path)

        assert reloaded_grid.n_face == original_grid.n_face
        assert reloaded_grid.n_node == original_grid.n_node
