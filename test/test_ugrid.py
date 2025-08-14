import os
import warnings
from pathlib import Path
import numpy.testing as nt
import pytest
import xarray as xr

import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

try:
    import constants
except ImportError:
    from . import constants

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_ne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug"
gridfile_RLL1deg = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
gridfile_RLL10deg_ne4 = current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"
gridfile_exo_ne8 = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"

def test_read_ugrid():
    """Reads a ugrid file."""
    uxgrid_ne30 = ux.open_grid(str(gridfile_ne30))
    uxgrid_RLL1deg = ux.open_grid(str(gridfile_RLL1deg))
    uxgrid_RLL10deg_ne4 = ux.open_grid(str(gridfile_RLL10deg_ne4))

    nt.assert_equal(uxgrid_ne30.node_lon.size, constants.NNODES_outCSne30)
    nt.assert_equal(uxgrid_RLL1deg.node_lon.size, constants.NNODES_outRLL1deg)
    nt.assert_equal(uxgrid_RLL10deg_ne4.node_lon.size, constants.NNODES_ov_RLL10deg_CSne4)

# Uncomment this test if you want to test OPeNDAP functionality
# def test_read_ugrid_opendap():
#     """Read an ugrid model from an OPeNDAP URL."""
#     url = "http://test.opendap.org:8080/opendap/ugrid/NECOFS_GOM3_FORECAST.nc"
#     try:
#         uxgrid_url = ux.open_grid(url, drop_variables="siglay")
#     except OSError:
#         warnings.warn(f'Could not connect to OPeNDAP server: {url}')
#     else:
#         assert isinstance(getattr(uxgrid_url, "node_lon"), xr.DataArray)
#         assert isinstance(getattr(uxgrid_url, "node_lat"), xr.DataArray)
#         assert isinstance(getattr(uxgrid_url, "face_node_connectivity"), xr.DataArray)

def test_to_xarray_ugrid():
    """Read an Exodus dataset and convert it to UGRID format using to_xarray."""
    ux_grid = ux.open_grid(gridfile_exo_ne8)
    xr_obj = ux_grid.to_xarray("UGRID")
    xr_obj.to_netcdf("ugrid_exo_csne8.nc")
    reloaded_grid = ux.open_grid("ugrid_exo_csne8.nc")
    # Check that the grid topology is perfectly preserved
    nt.assert_array_equal(ux_grid.face_node_connectivity.values,
                          reloaded_grid.face_node_connectivity.values)

    # Check that node coordinates are numerically close
    nt.assert_allclose(ux_grid.node_lon.values, reloaded_grid.node_lon.values)
    nt.assert_allclose(ux_grid.node_lat.values, reloaded_grid.node_lat.values)

    # Cleanup
    reloaded_grid._ds.close()
    del reloaded_grid
    os.remove("ugrid_exo_csne8.nc")

def test_standardized_dtype_and_fill():
    """Test to see if Mesh2_Face_Nodes uses the expected integer datatype and expected fill value."""
    ug_filenames = [
        current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30.ug",
        current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug",
        current_path / "meshfiles" / "ugrid" / "ov_RLL10deg_CSne4" / "ov_RLL10deg_CSne4.ug"
    ]

    grids_with_fill = [ux.open_grid(ug_filenames[1])]
    for grid in grids_with_fill:
        assert grid.face_node_connectivity.dtype == INT_DTYPE
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE
        assert INT_FILL_VALUE in grid.face_node_connectivity.values

    grids_without_fill = [ux.open_grid(ug_filenames[0]), ux.open_grid(ug_filenames[2])]
    for grid in grids_without_fill:
        assert grid.face_node_connectivity.dtype == INT_DTYPE
        assert grid.face_node_connectivity._FillValue == INT_FILL_VALUE

def test_standardized_dtype_and_fill_dask():
    """Test to see if Mesh2_Face_Nodes uses the expected integer datatype with dask chunking."""
    ug_filename = current_path / "meshfiles" / "ugrid" / "outRLL1deg" / "outRLL1deg.ug"
    ux_grid = ux.open_grid(ug_filename)

    assert ux_grid.face_node_connectivity.dtype == INT_DTYPE
    assert ux_grid.face_node_connectivity._FillValue == INT_FILL_VALUE
    assert INT_FILL_VALUE in ux_grid.face_node_connectivity.values

def test_encode_ugrid_copies_and_converts_bool_attr():
    """Test that encode_as('UGRID') returns a copy and converts boolean attrs to int."""
    import copy

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

    ds_orig = ds.copy(deep=True)
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
