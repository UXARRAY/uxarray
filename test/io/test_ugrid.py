import os
import warnings
import numpy as np
import numpy.testing as nt
import pytest
import xarray as xr

import uxarray as ux
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.conventions import ugrid


def test_edge_connectivity_dims_renamed_to_ugrid(gridpath):
    """The trailing dimension of the edge connectivities is renamed to 'two'.

    FESOM2 mesh diagnostics are the UGRID file in the test suite that ships edge
    connectivity, and it stores that dimension as 'n2'.
    """
    uxgrid = ux.open_grid(gridpath("ugrid", "fesom", "fesom.mesh.diag.nc"))

    for conn_name in ("edge_node_connectivity", "edge_face_connectivity"):
        assert conn_name in uxgrid._ds
        assert list(uxgrid._ds[conn_name].dims) == ugrid.CONNECTIVITY[conn_name]["dims"]

    assert "n2" not in uxgrid._ds.dims
    assert uxgrid._source_dims_dict["n2"] == "two"

    # Out of scope on purpose: the face connectivities keep sharing the source
    # file's single size-3 dimension, which is named after face_node_connectivity.
    assert uxgrid._ds["face_edge_connectivity"].dims == ("n_face", "n_max_face_nodes")


def test_edge_connectivity_dims_renamed_for_any_source_name(tmp_path):
    """The rename keys off the connectivity, not off a known set of dimension names.

    UGRID does not prescribe dimension names, and the spec's own examples call
    this axis 'Two'. Built here rather than added as a fixture so the test cannot
    be mistaken for something FESOM-specific.
    """
    edge_nodes = np.array([[0, 1], [1, 2], [2, 0], [2, 3], [3, 0]], dtype=INT_DTYPE)
    ds = xr.Dataset(
        {
            "Mesh2": xr.DataArray(
                np.int32(-1),
                attrs={
                    "cf_role": "mesh_topology",
                    "topology_dimension": np.int32(2),
                    "node_coordinates": "Mesh2_node_x Mesh2_node_y",
                    "face_node_connectivity": "Mesh2_face_nodes",
                    "edge_node_connectivity": "Mesh2_edge_nodes",
                    "face_dimension": "nMesh2_face",
                    "edge_dimension": "nMesh2_edge",
                },
            ),
            "Mesh2_node_x": xr.DataArray(
                np.array([0.0, 1.0, 1.0, 0.0]), dims="nMesh2_node"
            ),
            "Mesh2_node_y": xr.DataArray(
                np.array([0.0, 0.0, 1.0, 1.0]), dims="nMesh2_node"
            ),
            "Mesh2_face_nodes": xr.DataArray(
                np.array([[0, 1, 2], [0, 2, 3]], dtype=INT_DTYPE),
                dims=("nMesh2_face", "nMaxMesh2_face_nodes"),
                attrs={"cf_role": "face_node_connectivity", "start_index": 0},
            ),
            "Mesh2_edge_nodes": xr.DataArray(
                edge_nodes,
                dims=("nMesh2_edge", "Two"),
                attrs={"cf_role": "edge_node_connectivity", "start_index": 0},
            ),
        },
        attrs={"Conventions": "UGRID-1.0"},
    )

    path = tmp_path / "spec_ugrid_mesh.nc"
    ds.to_netcdf(path)

    uxgrid = ux.open_grid(path)

    assert uxgrid._ds["edge_node_connectivity"].dims == ("n_edge", "two")
    assert "Two" not in uxgrid._ds.dims
    nt.assert_array_equal(uxgrid._ds["edge_node_connectivity"].values, edge_nodes)


def test_read_ugrid(gridpath, mesh_constants):
    """Reads a ugrid file."""
    uxgrid_ne30 = ux.open_grid(str(gridpath("ugrid", "outCSne30", "outCSne30.ug")))
    uxgrid_RLL1deg = ux.open_grid(str(gridpath("ugrid", "outRLL1deg", "outRLL1deg.ug")))
    uxgrid_RLL10deg_ne4 = ux.open_grid(str(gridpath("ugrid", "ov_RLL10deg_CSne4", "ov_RLL10deg_CSne4.ug")))

    nt.assert_equal(uxgrid_ne30.node_lon.size, mesh_constants['NNODES_outCSne30'])
    nt.assert_equal(uxgrid_RLL1deg.node_lon.size, mesh_constants['NNODES_outRLL1deg'])
    nt.assert_equal(uxgrid_RLL10deg_ne4.node_lon.size, mesh_constants['NNODES_ov_RLL10deg_CSne4'])

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

def test_to_xarray_ugrid(gridpath):
    """Read an Exodus dataset and convert it to UGRID format using to_xarray."""
    ux_grid = ux.open_grid(gridpath("exodus", "outCSne8", "outCSne8.g"))
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
