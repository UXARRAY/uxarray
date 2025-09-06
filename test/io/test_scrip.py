import os
import xarray as xr
import warnings
import numpy.testing as nt
import pytest
from pathlib import Path

import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *
import constants

# Define grid file paths
gridfile_ne30 = OUTCSNE30_GRID
gridfile_RLL1deg = OUTRLL1DEG_GRID
gridfile_RLL10deg_ne4 = OV_RLL10DEG_CSNE4_GRID
gridfile_exo_ne8 = EXODUS_OUTCSNE8
gridfile_scrip = SCRIP_OUTCSNE8

def test_read_ugrid():
    """Reads a ugrid file."""
    uxgrid_ne30 = ux.open_grid(str(gridfile_ne30))
    uxgrid_RLL1deg = ux.open_grid(str(gridfile_RLL1deg))
    uxgrid_RLL10deg_ne4 = ux.open_grid(str(gridfile_RLL10deg_ne4))

    nt.assert_equal(uxgrid_ne30.node_lon.size, constants.NNODES_outCSne30)
    nt.assert_equal(uxgrid_RLL1deg.node_lon.size, constants.NNODES_outRLL1deg)
    nt.assert_equal(uxgrid_RLL10deg_ne4.node_lon.size, constants.NNODES_ov_RLL10deg_CSne4)

# TODO: UNCOMMENT
# def test_read_ugrid_opendap():
#     """Read an ugrid model from an OPeNDAP URL."""
#     try:
#         url = "http://test.opendap.org:8080/opendap/ugrid/NECOFS_GOM3_FORECAST.nc"
#         uxgrid_url = ux.open_grid(url, drop_variables="siglay")
#     except OSError:
#         warnings.warn(f'Could not connect to OPeNDAP server: {url}')
#         pass
#     else:
#         assert isinstance(getattr(uxgrid_url, "node_lon"), xr.DataArray)
#         assert isinstance(getattr(uxgrid_url, "node_lat"), xr.DataArray)
#         assert isinstance(getattr(uxgrid_url, "face_node_connectivity"), xr.DataArray)

def test_to_xarray_ugrid():
    """Read an Exodus dataset and convert it to UGRID format using to_xarray."""
    ux_grid = ux.open_grid(gridfile_scrip)
    xr_obj = ux_grid.to_xarray("UGRID")
    xr_obj.to_netcdf("scrip_ugrid_csne8.nc")
    reloaded_grid = ux.open_grid("scrip_ugrid_csne8.nc")
    # Check that the grid topology is perfectly preserved
    nt.assert_array_equal(ux_grid.face_node_connectivity.values,
                          reloaded_grid.face_node_connectivity.values)

    # Check that node coordinates are numerically close
    nt.assert_allclose(ux_grid.node_lon.values, reloaded_grid.node_lon.values)
    nt.assert_allclose(ux_grid.node_lat.values, reloaded_grid.node_lat.values)

    # Cleanup
    reloaded_grid._ds.close()
    del reloaded_grid
    os.remove("scrip_ugrid_csne8.nc")

def test_standardized_dtype_and_fill():
    """Test to see if Mesh2_Face_Nodes uses the expected integer datatype
    and expected fill value as set in constants.py."""
    ug_filename1 = OUTCSNE30_GRID
    ug_filename2 = OUTRLL1DEG_GRID
    ug_filename3 = OV_RLL10DEG_CSNE4_GRID

    ux_grid1 = ux.open_grid(ug_filename1)
    ux_grid2 = ux.open_grid(ug_filename2)
    ux_grid3 = ux.open_grid(ug_filename3)

    # Check for correct dtype and fill value
    grids_with_fill = [ux_grid2]
    for grid in grids_with_fill:
        assert grid.face_node_connectivity.dtype == INT_DTYPE
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE
        assert INT_FILL_VALUE in grid.face_node_connectivity.values

    grids_without_fill = [ux_grid1, ux_grid3]
    for grid in grids_without_fill:
        assert grid.face_node_connectivity.dtype == INT_DTYPE
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE

def test_standardized_dtype_and_fill_dask():
    """Test to see if Mesh2_Face_Nodes uses the expected integer datatype
    and expected fill value as set in constants.py with dask chunking."""
    ug_filename = OUTRLL1DEG_GRID
    ux_grid = ux.open_grid(ug_filename)

    assert ux_grid.face_node_connectivity.dtype == INT_DTYPE
    assert ux_grid.face_node_connectivity._FillValue == INT_FILL_VALUE
    assert INT_FILL_VALUE in ux_grid.face_node_connectivity.values
