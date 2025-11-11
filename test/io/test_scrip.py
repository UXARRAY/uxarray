import os
import xarray as xr
import warnings
import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.io._scrip import _detect_multigrid


def test_read_ugrid(gridpath, mesh_constants):
    """Reads a ugrid file."""
    uxgrid_ne30 = ux.open_grid(str(gridpath("ugrid", "outCSne30", "outCSne30.ug")))
    uxgrid_RLL1deg = ux.open_grid(str(gridpath("ugrid", "outRLL1deg", "outRLL1deg.ug")))
    uxgrid_RLL10deg_ne4 = ux.open_grid(str(gridpath("ugrid", "ov_RLL10deg_CSne4", "ov_RLL10deg_CSne4.ug")))

    nt.assert_equal(uxgrid_ne30.node_lon.size, mesh_constants['NNODES_outCSne30'])
    nt.assert_equal(uxgrid_RLL1deg.node_lon.size, mesh_constants['NNODES_outRLL1deg'])
    nt.assert_equal(uxgrid_RLL10deg_ne4.node_lon.size, mesh_constants['NNODES_ov_RLL10deg_CSne4'])

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

def test_to_xarray_ugrid(gridpath):
    """Read an Exodus dataset and convert it to UGRID format using to_xarray."""
    ux_grid = ux.open_grid(gridpath("scrip", "outCSne8", "outCSne8.nc"))
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


def test_oasis_multigrid_format_detection():
    """Detect OASIS-style multi-grid naming."""
    ds = xr.Dataset()
    ds["ocn.cla"] = xr.DataArray(np.random.rand(100, 4), dims=["nc_ocn", "nv_ocn"])
    ds["ocn.clo"] = xr.DataArray(np.random.rand(100, 4), dims=["nc_ocn", "nv_ocn"])
    ds["atm.cla"] = xr.DataArray(np.random.rand(200, 4), dims=["nc_atm", "nv_atm"])
    ds["atm.clo"] = xr.DataArray(np.random.rand(200, 4), dims=["nc_atm", "nv_atm"])

    format_type, grids = _detect_multigrid(ds)
    assert format_type == "multi_scrip"
    assert set(grids.keys()) == {"ocn", "atm"}


def test_open_multigrid_with_masks(gridpath):
    """Load OASIS multi-grids with masks applied."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    mask_file = gridpath("scrip", "oasis", "masks.nc")

    grids = ux.open_multigrid(grid_file, mask_filename=mask_file)
    assert grids["ocn"].n_face == 8
    assert grids["atm"].n_face == 20

    ocean_only = ux.open_multigrid(
        grid_file, gridnames=["ocn"], mask_filename=mask_file
    )
    assert set(ocean_only.keys()) == {"ocn"}
    assert ocean_only["ocn"].n_face == 8

    grid_names = ux.list_grid_names(grid_file)
    assert set(grid_names) == {"ocn", "atm"}
