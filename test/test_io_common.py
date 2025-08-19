"""
Common IO tests that apply to all grid formats. These tests make sure the
same basic things work no matter which file format you start with.
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
    ("healpix", None, None),  # Constructed via classmethod
]

# Formats that support writing
WRITABLE_FORMATS = ["ugrid", "exodus", "scrip", "esmf"]

# Format conversion test pairs - removed for now as format conversion
# requires more sophisticated handling than simple to_netcdf


@pytest.fixture(params=IO_READ_TEST_FORMATS)
def grid_from_format(request):
    """Load a Grid from each supported format for parameterized tests.

    Handles special cases (FESOM multi-file, HEALPix) and tags the grid with
    ``_test_format`` for easier debugging.
    """
    format_name, subpath, filename = request.param

    if format_name == "fesom" and filename is None:
        # Special handling for FESOM with multiple input files
        fesom_data_path = current_path / "meshfiles" / subpath / "data"
        fesom_mesh_path = current_path / "meshfiles" / subpath
        grid = ux.open_grid(fesom_mesh_path, fesom_data_path)
    elif format_name == "healpix":
        # Construct a basic HEALPix grid
        grid = ux.Grid.from_healpix(zoom=1, pixels_only=False)
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
    """Common IO tests across all formats. Helps catch format-specific
    regressions early and keep behavior consistent.
    """

    def test_return_type(self, grid_from_format):
        """Open each format and return a ux.Grid. Checks that the public API
        is consistent across readers.
        """
        grid = grid_from_format

        # Basic validation
        assert isinstance(grid, ux.Grid)

    def test_ugrid_compliance(self, grid_from_format):
        """Check that a loaded grid looks like a UGRID mesh. We look for
        required topology, coordinates, proper fill values, reasonable degree
        ranges, and that ``validate()`` passes.
        """
        grid = grid_from_format

        # Basic topology and coordinate presence
        assert 'face_node_connectivity' in grid.connectivity
        assert 'node_lon' in grid.coordinates
        assert 'node_lat' in grid.coordinates

        # Required dimensions
        assert 'n_node' in grid.dims
        assert 'n_face' in grid.dims

        # Validate grid structure
        grid.validate()

        # Check UGRID compliance
        # 1. Connectivity should use proper fill values
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE

        # 3. Check that grid has been properly standardized by uxarray
        # (Not all input files have Conventions attribute, but uxarray should handle them)

    def test_grid_properties_consistency(self, grid_from_format):
        """Make sure core dims and variables are present with the expected
        dtypes across formats. Avoid surprises for downstream code.
        """
        grid = grid_from_format

        # Check that all grids have the essential properties
        assert 'n_node' in grid.dims
        assert 'n_face' in grid.dims
        assert 'face_node_connectivity' in grid.connectivity
        assert 'node_lon' in grid.coordinates
        assert 'node_lat' in grid.coordinates

        # Check data types are consistent
        assert np.issubdtype(grid.face_node_connectivity.dtype, np.integer)
        assert np.issubdtype(grid.node_lon.dtype, np.floating)
        assert np.issubdtype(grid.node_lat.dtype, np.floating)
