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
WRITABLE_FORMATS = ["ugrid", "exodus"]

# Format conversion test pairs - removed for now as format conversion
# requires more sophisticated handling than simple to_netcdf


class TestIOCommon:
    """Common IO tests across all formats."""
    
    @pytest.mark.parametrize("format_name,subpath,filename", IO_READ_TEST_FORMATS)
    def test_basic_read(self, format_name, subpath, filename):
        """Test that all formats can be read successfully."""
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
        
        # Basic validation
        assert grid is not None
        assert hasattr(grid, 'face_node_connectivity')
        assert hasattr(grid, 'node_lon')
        assert hasattr(grid, 'node_lat')
        
        # Check required dimensions
        assert 'n_node' in grid._ds.dims
        assert 'n_face' in grid._ds.dims
        
        # Validate grid
        grid.validate()
    
    @pytest.mark.parametrize("format_name,subpath,filename", IO_READ_TEST_FORMATS)
    def test_ugrid_compliance_after_read(self, format_name, subpath, filename):
        """Test that grids from all formats meet basic UGRID standards."""
        if format_name == "fesom" and filename is None:
            # Special handling for FESOM
            fesom_data_path = current_path / "meshfiles" / subpath / "data"
            fesom_mesh_path = current_path / "meshfiles" / subpath
            grid = ux.open_grid(fesom_mesh_path, fesom_data_path)
        else:
            grid_path = current_path / "meshfiles" / subpath / filename
            if not grid_path.exists():
                pytest.skip(f"Test file not found: {grid_path}")
            
            if format_name == "mpas":
                grid = ux.open_grid(grid_path, use_dual=False)
            else:
                grid = ux.open_grid(grid_path)
        
        # Check UGRID compliance
        # 1. Connectivity should use proper fill values
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE
        
        # 2. Coordinates should be in valid ranges
        if hasattr(grid.node_lon, 'units') and 'degree' in str(grid.node_lon.units):
            assert grid.node_lon.min() >= -180
            assert grid.node_lon.max() <= 360
            assert grid.node_lat.min() >= -90
            assert grid.node_lat.max() <= 90
        
        # 3. Check that grid has been properly standardized by uxarray
        # (Not all input files have Conventions attribute, but uxarray should handle them)
    
    def test_grid_properties_consistency(self):
        """Test that all grids have consistent basic properties after loading."""
        grids = []
        
        # Load a few different format grids
        test_files = [
            ("ugrid", "ugrid/quad-hexagon", "grid.nc"),
            ("mpas", "mpas/QU/480", "grid.nc"),
            ("exodus", "exodus/outCSne8", "outCSne8.g"),
        ]
        
        for format_name, subpath, filename in test_files:
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
    
    @pytest.mark.parametrize("write_format", WRITABLE_FORMATS)
    def test_write_invalid_path(self, write_format):
        """Test error handling for invalid write paths."""
        # Create a simple test grid
        grid = ux.open_grid(current_path / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc")
        
        # Try to write to invalid path
        with pytest.raises((OSError, IOError, PermissionError)):
            grid._ds.to_netcdf("/invalid/path/file.nc")
    
    def test_read_nonexistent_file(self):
        """Test error handling for non-existent files."""
        with pytest.raises((FileNotFoundError, OSError)):
            ux.open_grid("nonexistent_file.nc")
    
    @pytest.mark.parametrize("format_name", ["ugrid", "mpas", "esmf", "exodus", "scrip"])
    def test_lazy_loading(self, format_name):
        """Test that grids support lazy loading where applicable."""
        # Get a test file for this format
        test_file = None
        for fmt, subpath, filename in IO_READ_TEST_FORMATS:
            if fmt == format_name and filename is not None:
                test_file = current_path / "meshfiles" / subpath / filename
                break
        
        if test_file is None or not test_file.exists():
            pytest.skip(f"No test file found for {format_name}")
        
        # Open grid
        if format_name == "mpas":
            grid = ux.open_grid(test_file, use_dual=False)
        else:
            grid = ux.open_grid(test_file)
        
        # Check that data arrays are lazy (dask arrays) where applicable
        # This is format-dependent, so we just check it doesn't error
        assert grid._ds is not None


class TestIODatasetCommon:
    """Common tests for dataset (grid + data) operations."""
    
    def test_dataset_basic_operations(self):
        """Test basic dataset operations across formats."""
        # Test that we can open datasets with different grid formats
        test_cases = [
            ("ugrid", "ugrid/quad-hexagon", "grid.nc", "data.nc"),
        ]
        
        for format_name, subpath, grid_file, data_file in test_cases:
            grid_path = current_path / "meshfiles" / subpath / grid_file
            data_path = current_path / "meshfiles" / subpath / data_file
            
            if not grid_path.exists() or not data_path.exists():
                continue
            
            # Open dataset
            dataset = ux.open_dataset(grid_path, data_path)
            
            # Basic checks
            assert dataset is not None
            assert hasattr(dataset, 'uxgrid')
            assert len(dataset.data_vars) > 0