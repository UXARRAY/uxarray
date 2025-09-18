import os
import xarray as xr
import warnings
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


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
