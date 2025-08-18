"""
Common IO tests that apply to all grid formats.

This module tests functionality that should work across all supported formats:
- Basic read/write operations
- Format conversions and round-trips
- UGRID compliance
- Common error handling
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

# Define all testable format combinations
# Format: (format_type, subpath, filename)
IO_READ_TEST_FORMATS = [
    ("ugrid", "ugrid/quad-hexagon", "grid.nc"),
    ("ugrid", "ugrid/outCSne30", "outCSne30.ug"),
    ("ugrid", "ugrid/outRLL1deg", "outRLL1deg.ug"),
    ("mpas", "mpas/QU/480", "grid.nc"),
    ("esmf", "esmf/ne30", "ne30pg3.grid.nc"),
    ("exodus", "exodus/outCSne8", "outCSne8.g"),
    ("exodus", "exodus/mixed", "mixed.exo"),
    ("scrip", "scrip/outCSne8", "outCSne8.nc"),
    ("icon", "icon/R02B04", "icon_grid_0010_R02B04_G.nc"),
    ("fesom", "fesom/pi", None),  # Special case - multiple files
]

# Formats that support writing
WRITABLE_FORMATS = ["ugrid", "exodus", "scrip", "esmf"]

# Format conversion test pairs - removed for now as format conversion
# requires more sophisticated handling than simple to_netcdf


@pytest.fixture(params=IO_READ_TEST_FORMATS)
def grid_from_format(request):
    """Fixture that loads grids from all supported formats."""
    format_name, subpath, filename = request.param

    if format_name == "fesom" and filename is None:
        # Special handling for FESOM with multiple input files
        fesom_data_path = current_path / "meshfiles" / subpath / "data"
        fesom_mesh_path = current_path / "meshfiles" / subpath
        grid = ux.open_grid(fesom_mesh_path, fesom_data_path)
    else:
        grid_path = current_path / "meshfiles" / subpath / filename
        if not grid_path.exists():
            pytest.skip(f"Test file not found: {grid_path}")

        # Handle special cases
        if format_name == "mpas":
            grid = ux.open_grid(grid_path, use_dual=False)
        else:
            grid = ux.open_grid(grid_path)

    # Add format info to the grid for test identification
    grid._test_format = format_name
    return grid


class TestIOCommon:
    """Common IO tests across all formats."""

    def test_return_type(self, grid_from_format):
        """Test that all formats can be read successfully."""
        grid = grid_from_format

        # Basic validation
        assert isinstance(grid, ux.Grid)
        assert hasattr(grid, 'face_node_connectivity')
        assert hasattr(grid, 'node_lon')
        assert hasattr(grid, 'node_lat')

        # Check required dimensions
        assert 'n_node' in grid.dims
        assert 'n_face' in grid.dims

        # Validate grid
        grid.validate()

    def test_ugrid_compliance(self, grid_from_format):
        """Test that grids from all formats meet basic UGRID standards."""
        grid = grid_from_format

        # Check UGRID compliance
        # 1. Connectivity should use proper fill values
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE

        # 2. Coordinates should be in valid ranges
        if hasattr(grid.node_lon, 'units') and 'degree' in str(grid.node_lon.units):
            assert grid.node_lon.min() >= -180
            assert grid.node_lon.max() <= 180
            assert grid.node_lat.min() >= -90
            assert grid.node_lat.max() <= 90

        # 3. Check that grid has been properly standardized by uxarray
        # (Not all input files have Conventions attribute, but uxarray should handle them)

    def test_grid_properties_consistency(self):
        """Test that all grids have consistent basic properties after loading."""
        grids = []

        # Load all different format grids
        for format_name, subpath, filename in IO_READ_TEST_FORMATS:
            if filename is None:  # Special case for fesom
                # Skip fesom for now as it requires multiple files
                continue

            grid_path = current_path / "meshfiles" / subpath / filename
            if not grid_path.exists():
                continue

            if format_name == "mpas":
                grid = ux.open_grid(grid_path, use_dual=False)
            else:
                grid = ux.open_grid(grid_path)

            # Check that all grids have the essential properties
            assert hasattr(grid, 'n_node')
            assert hasattr(grid, 'n_face')
            assert hasattr(grid, 'face_node_connectivity')
            assert hasattr(grid, 'node_lon')
            assert hasattr(grid, 'node_lat')

            # Check data types are consistent
            assert grid.face_node_connectivity.dtype in [np.int32, np.int64, INT_DTYPE]
            assert grid.node_lon.dtype in [np.float32, np.float64]
            assert grid.node_lat.dtype in [np.float32, np.float64]



    def test_lazy_loading(self, grid_from_format):
        """Test that grids support lazy loading where applicable."""
        grid = grid_from_format

        assert grid._ds is not None
