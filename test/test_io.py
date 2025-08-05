"""
Unified IO testing module for uxarray.

This module provides comprehensive testing for all supported IO formats including:
- UGRID, Exodus, ESMF, SCRIP, MPAS, ICON, FESOM
- Reading, writing, and round-trip consistency tests
- Edge cases and error handling
- Format-specific functionality

Follows pytest best practices with parametrized tests and fixture-based design.
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
from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE, INT_FILL_VALUE

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))


# Test data fixtures
@pytest.fixture(scope="session")
def test_data_paths():
    """Centralized test data paths for all formats."""
    base_path = current_path / "meshfiles"
    return {
        "ugrid": {
            "ne30": base_path / "ugrid" / "outCSne30" / "outCSne30.ug",
            "rll1deg": base_path / "ugrid" / "outRLL1deg" / "outRLL1deg.ug",
            "rll10deg_ne4": base_path / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"
        },
        "exodus": {
            "ne8": base_path / "exodus" / "outCSne8" / "outCSne8.g",
            "mixed": base_path / "exodus" / "mixed" / "mixed.exo"
        },
        "esmf": {
            "ne30_grid": base_path / "esmf" / "ne30" / "ne30pg3.grid.nc",
            "ne30_data": base_path / "esmf" / "ne30" / "ne30pg3.data.nc"
        },
        "scrip": {
            "ne8": base_path / "scrip" / "outCSne8" / "outCSne8.nc"
        },
        "mpas": {
            "qu_grid": base_path / "mpas" / "QU" / "mesh.QU.1920km.151026.nc",
            "qu_ocean": base_path / "mpas" / "QU" / "oQU480.231010.nc"
        },
        "icon": {
            "r02b04": base_path / "icon" / "R02B04" / "icon_grid_0010_R02B04_G.nc"
        },
        "fesom": {
            "ugrid_diag": base_path / "ugrid" / "fesom" / "fesom.mesh.diag.nc",
            "ascii": base_path / "fesom" / "pi",
            "netcdf": base_path / "fesom" / "soufflet-netcdf" / "grid.nc"
        }
    }


@pytest.fixture
def temp_output_dir():
    """Provide a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Parametrized test configurations
READABLE_FORMATS = [
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

WRITABLE_FORMATS = ["UGRID", "Exodus", "ESMF"]

ROUND_TRIP_CONFIGS = [
    ("ugrid", "ne30", "UGRID"),
    ("exodus", "mixed", "UGRID"),
    ("exodus", "mixed", "Exodus"),
    ("esmf", "ne30_grid", "ESMF"),
    ("scrip", "ne8", "UGRID")
]


class TestIOReaders:
    """Test reading functionality for all supported formats."""

    @pytest.mark.parametrize("format_name,data_key", READABLE_FORMATS)
    def test_read_grid_basic(self, test_data_paths, format_name, data_key):
        """Test basic grid reading for all formats."""
        grid_path = test_data_paths[format_name][data_key]
        if not grid_path.exists():
            pytest.skip(f"Test file not found: {grid_path}")

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

        # Check connectivity array properties
        # Allow both int32 and int64 dtypes as different formats may use different sizes
        assert grid.face_node_connectivity.dtype in [INT_DTYPE, np.int32, np.int64]
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE

        # Check coordinate consistency
        assert len(grid.node_lon) == len(grid.node_lat)
        assert grid.node_lon.size > 0
        assert grid.node_lat.size > 0

        # Validate grid
        grid.validate()

    @pytest.mark.parametrize("format_name,data_key", [
        ("ugrid", "ne30"),
        ("ugrid", "rll1deg"),
        ("ugrid", "rll10deg_ne4")
    ])
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


class TestIOWriters:
    """Test writing functionality for supported formats."""

    @pytest.mark.parametrize("format_name", WRITABLE_FORMATS)
    def test_write_format(self, test_data_paths, temp_output_dir, format_name):
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


class TestIORoundTrip:
    """Test round-trip consistency for read-write operations."""

    @pytest.mark.parametrize("format_name,data_key,output_format", ROUND_TRIP_CONFIGS)
    def test_round_trip_consistency(self, test_data_paths, temp_output_dir, format_name, data_key, output_format):
        """Test that data survives round-trip conversion."""
        input_path = test_data_paths[format_name][data_key]
        if not input_path.exists():
            pytest.skip(f"Test file not found: {input_path}")

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

    def test_esmf_round_trip_detailed(self, test_data_paths, temp_output_dir):
        """Test detailed ESMF round-trip with specific validation."""
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


class TestIODatasets:
    """Test dataset (grid + data) functionality."""

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

    def test_fesom_dataset_loading(self, test_data_paths):
        """Test FESOM dataset loading with multiple data files."""
        grid_paths = ["ugrid_diag", "ascii"]

        for grid_key in grid_paths:
            grid_path = test_data_paths["fesom"][grid_key]
            if not grid_path.exists():
                continue

            if grid_key == "ascii":
                # Test with single data file
                data_path = grid_path / "data" / "sst.fesom.1985.nc"
                if data_path.exists():
                    dataset = ux.open_dataset(grid_path, data_path)
                    assert "n_node" in dataset.dims
                    assert len(dataset) == 1

                # Test with multiple data files
                data_pattern = str(grid_path / "data" / "*.nc")
                data_files = glob.glob(data_pattern)
                if data_files:
                    dataset = ux.open_mfdataset(grid_path, data_files)
                    assert "n_node" in dataset.dims
                    assert "n_face" in dataset.dims
                    assert len(dataset) >= 1
            else:
                # Just test grid loading for other formats
                grid = ux.open_grid(grid_path)
                assert grid is not None


class TestIOEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        with pytest.raises((FileNotFoundError, OSError)):
            ux.open_grid("nonexistent_file.nc")

    def test_corrupted_file_handling(self):
        """Test handling of corrupted files."""
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
            f.write(b"corrupted data")
            f.flush()

            with pytest.raises(Exception):  # Should raise some kind of parsing error
                ux.open_grid(f.name)

            os.unlink(f.name)

    @pytest.mark.parametrize("format_name", WRITABLE_FORMATS)
    def test_empty_grid_handling(self, format_name):
        """Test handling of empty or minimal grids."""
        # Create minimal valid grid
        ds = xr.Dataset(
            {
                "node_lon": (("n_node",), [0.0, 1.0, 0.5]),
                "node_lat": (("n_node",), [0.0, 0.0, 1.0]),
                "face_node_connectivity": (("n_face", "n_max_face_nodes"), [[0, 1, 2]])
            }
        )

        grid = ux.Grid(ds)

        # Should be able to convert to any writable format
        result = grid.to_xarray(format_name)
        assert isinstance(result, xr.Dataset)

    def test_standardized_dtype_and_fill_values(self, test_data_paths):
        """Test that all formats use standardized dtype and fill values."""
        test_files = [
            ("ugrid", "ne30"),
            ("ugrid", "rll1deg"),
            ("exodus", "mixed"),
            ("scrip", "ne8")
        ]

        for format_name, data_key in test_files:
            grid_path = test_data_paths[format_name][data_key]
            if not grid_path.exists():
                continue

            grid = ux.open_grid(grid_path)

            # Check dtype and fill value
            # Allow both int32 and int64 dtypes as different formats may use different sizes
            assert grid.face_node_connectivity.dtype in [INT_DTYPE, np.int32, np.int64]
            assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE


class TestFormatSpecific:
    """Format-specific tests that don't fit the general patterns."""

    def test_mpas_dual_mesh(self, test_data_paths):
        """Test MPAS dual mesh functionality."""
        grid_path = test_data_paths["mpas"]["qu_grid"]
        if not grid_path.exists():
            pytest.skip("MPAS test file not found")

        primal_grid = ux.open_grid(grid_path, use_dual=False)
        dual_grid = ux.open_grid(grid_path, use_dual=True)

        assert primal_grid is not None
        assert dual_grid is not None

        # Dual mesh should have different structure
        assert dual_grid.face_node_connectivity.shape[1] == 3  # Triangular faces for dual

    def test_mpas_face_areas(self, test_data_paths):
        """Test MPAS face area parsing."""
        for data_key in ["qu_grid", "qu_ocean"]:
            grid_path = test_data_paths["mpas"][data_key]
            if not grid_path.exists():
                continue

            primal_grid = ux.open_grid(grid_path, use_dual=False)
            dual_grid = ux.open_grid(grid_path, use_dual=True)

            assert "face_areas" in primal_grid._ds
            assert "face_areas" in dual_grid._ds

    def test_mpas_distance_units(self, test_data_paths):
        """Test MPAS distance unit conversion."""
        grid_path = test_data_paths["mpas"]["qu_ocean"]
        if not grid_path.exists():
            pytest.skip("MPAS ocean test file not found")

        grid = ux.open_grid(grid_path)

        assert "edge_node_distances" in grid._ds
        assert "edge_face_distances" in grid._ds

    def test_fesom_format_comparison(self, test_data_paths):
        """Test consistency between FESOM ASCII and NetCDF formats."""
        ascii_path = test_data_paths["fesom"]["ascii"]
        ugrid_path = test_data_paths["fesom"]["ugrid_diag"]

        if not (ascii_path.exists() and ugrid_path.exists()):
            pytest.skip("FESOM test files not found")

        grid_ascii = ux.open_grid(ascii_path)
        grid_ugrid = ux.open_grid(ugrid_path)

        assert grid_ascii.n_face == grid_ugrid.n_face
        assert grid_ascii.n_node == grid_ugrid.n_node

        assert_array_equal(
            grid_ascii.face_node_connectivity.values,
            grid_ugrid.face_node_connectivity.values
        )

    def test_exodus_mixed_face_types(self, test_data_paths, temp_output_dir):
        """Test Exodus files with mixed face types (triangles and quadrilaterals)."""
        grid_path = test_data_paths["exodus"]["mixed"]
        if not grid_path.exists():
            pytest.skip("Exodus mixed test file not found")

        grid = ux.open_grid(grid_path)

        # Convert to both UGRID and Exodus formats
        ugrid_obj = grid.to_xarray("UGRID")
        exo_obj = grid.to_xarray("Exodus")

        ugrid_file = temp_output_dir / "test_ugrid.nc"
        exo_file = temp_output_dir / "test_exo.exo"

        ugrid_obj.to_netcdf(ugrid_file)
        exo_obj.to_netcdf(exo_file)

        ugrid_reloaded = ux.open_grid(ugrid_file)
        exodus_reloaded = ux.open_grid(exo_file)

        # Face node connectivity comparison
        assert_array_equal(
            ugrid_reloaded.face_node_connectivity.values,
            grid.face_node_connectivity.values
        )
        assert_array_equal(
            grid.face_node_connectivity.values,
            exodus_reloaded.face_node_connectivity.values
        )

        # Node coordinates comparison
        assert_array_equal(ugrid_reloaded.node_lon.values, grid.node_lon.values)
        assert_array_equal(grid.node_lon.values, exodus_reloaded.node_lon.values)
        assert_array_equal(ugrid_reloaded.node_lat.values, grid.node_lat.values)

    def test_icon_basic_functionality(self, test_data_paths):
        """Test basic ICON grid functionality."""
        grid_path = test_data_paths["icon"]["r02b04"]
        if not grid_path.exists():
            pytest.skip("ICON test file not found")

        # Test grid reading
        grid = ux.open_grid(grid_path)
        assert grid is not None

        # Test dataset reading
        dataset = ux.open_dataset(grid_path, grid_path)
        assert dataset is not None

    def test_vertex_grid_creation(self):
        """Test creating grid from vertices (Exodus format)."""
        verts = [[[0, 0], [2, 0], [0, 2], [2, 2]]]
        grid = ux.open_grid(verts)
        assert grid is not None
        assert grid.n_face == 1
        assert grid.n_node == 4


class TestIOPerformance:
    """Performance-related IO tests."""

    @pytest.mark.parametrize("format_name,data_key", [
        ("ugrid", "ne30"),
        ("exodus", "ne8"),
        ("mpas", "qu_grid")
    ])
    def test_lazy_loading(self, test_data_paths, format_name, data_key):
        """Test that grid loading is reasonably fast (lazy where possible)."""
        import time

        grid_path = test_data_paths[format_name][data_key]
        if not grid_path.exists():
            pytest.skip(f"Test file not found: {grid_path}")

        start_time = time.time()

        if format_name == "mpas":
            grid = ux.open_grid(grid_path, use_dual=False)
        else:
            grid = ux.open_grid(grid_path)

        load_time = time.time() - start_time

        # Basic validation that grid loaded successfully
        assert grid is not None
        assert hasattr(grid, 'node_lon')

        # Loading should complete in reasonable time (adjust threshold as needed)
        # This is more of a smoke test than a strict performance requirement
        assert load_time < 30.0, f"Grid loading took {load_time:.2f}s, which seems excessive"


# Cleanup functions for temporary files
def pytest_runtest_teardown(item, nextitem):
    """Clean up any temporary files after each test."""
    # Remove any temporary files that might have been created
    temp_files = [
        "test_ugrid.nc",
        "test_exo.exo",
        "scrip_ugrid_csne8.nc",
        "ugrid_exo_csne8.nc"
    ]

    for temp_file in temp_files:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError:
                pass  # File might be in use or already removed
