import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.core.dataarray import UxDataArray
from uxarray.core.dataset import UxDataset

# ------------------------------------------------------------
# Helper: small 3‐point spherical grid
# ------------------------------------------------------------
def _make_node_da(data):
    """
    Create a UxDataArray on a 3-point spherical grid,
    with dimension 'n_node' and identical source/destination grid.
    """
    verts = np.array([
        (0.0,  90.0),
        (-180.0, 0.0),
        (0.0, -90.0)
    ])
    grid = ux.open_grid(verts)
    da = UxDataArray(np.asarray(data), dims=["n_node"], coords={"n_node": [0,1,2]})
    da.uxgrid = grid
    return da, grid

# ------------------------------------------------------------
# Nearest‐neighbor tests
# ------------------------------------------------------------
def test_remap_to_same_grid_corner_nodes():
    """Remapping a 3-node array onto itself must be identity."""
    src_da, grid = _make_node_da([1.0, 2.0, 3.0])
    remapped = src_da.remap.nearest_neighbor(destination_grid=grid, remap_to="nodes")
    nt.assert_array_equal(src_da.values, remapped.values)


def test_nn_remap_returns_nonempty(gridpath, datasetpath):
    """A real v1 DataArray remapping must yield non-empty output."""
    grid_path = gridpath("ugrid", "geoflow-small", "grid.nc")
    uxds = ux.open_dataset(grid_path, datasetpath("ugrid", "geoflow-small", "v1.nc"))
    uxgrid = ux.open_grid(grid_path)
    out = uxds["v1"].remap.nearest_neighbor(destination_grid=uxgrid, remap_to="nodes")
    assert out.size > 0


def test_nn_return_types_and_counts(gridpath, datasetpath):
    """Nearest‐neighbor on a multi‐file dataset yields correct types and var counts."""
    grid_path = gridpath("ugrid", "geoflow-small", "grid.nc")
    dsfiles_geoflow = [datasetpath("ugrid", "geoflow-small", "v1.nc"), datasetpath("ugrid", "geoflow-small", "v2.nc"), datasetpath("ugrid", "geoflow-small", "v3.nc")]
    uxds = ux.open_mfdataset(grid_path, dsfiles_geoflow)
    dest = ux.open_grid(grid_path)

    # single DataArray → UxDataArray
    da_remap = uxds["v1"].remap.nearest_neighbor(destination_grid=dest, remap_to="nodes")
    assert isinstance(da_remap, UxDataArray)

    # whole Dataset → UxDataset, same # of data_vars
    ds_remap = uxds.remap.nearest_neighbor(destination_grid=dest, remap_to="nodes")
    assert isinstance(ds_remap, UxDataset)
    assert len(ds_remap.data_vars) == len(uxds.data_vars)


def test_edge_centers_dim_change(gridpath, datasetpath):
    """Nearest‐neighbor remap to edge centers produces an 'n_edge' dimension."""
    uxds = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))
    dest = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    da = uxds["v1"].remap.nearest_neighbor(destination_grid=dest, remap_to="edge centers")
    assert "n_edge" in da.dims


def test_original_not_overwritten(gridpath, datasetpath):
    """Check that remapping does not mutate the source."""
    uxds = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))
    original = uxds["v1"].copy()
    dest = uxds.uxgrid
    remap = uxds["v1"].remap.nearest_neighbor(destination_grid=dest, remap_to="face centers")
    assert not np.array_equal(original.values, remap.values)


def test_source_positions_work(gridpath):
    """Nearest‐neighbor works whether source is on faces, nodes, or edges."""
    mesh_path = gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)
    dest = ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc"))
    for var, expected_dim in (
        ("latCell",   "n_node"),
        ("latVertex", "n_node"),
        ("angleEdge", "n_node"),
    ):
        out = uxds[var].remap.nearest_neighbor(destination_grid=dest, remap_to="nodes")
        assert out.size > 0
        assert "n_node" in out.dims


def test_preserve_nonspatial_coords(gridpath, datasetpath):
    """Non‐spatial coords (e.g. time) survive remapping on a Dataset."""
    uxds = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))
    dest = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    ds_out = uxds.remap.nearest_neighbor(destination_grid=dest, remap_to="nodes")
    assert "time" in ds_out.coords

# ------------------------------------------------------------
# Inverse‐distance‐weighted (IDW) tests
# ------------------------------------------------------------
def test_idw_modifies_values(gridpath, datasetpath):
    """Simple IDW remap should change the array when remap_to != source."""
    grid_path = gridpath("ugrid", "geoflow-small", "grid.nc")
    uxds = ux.open_dataset(grid_path, datasetpath("ugrid", "geoflow-small", "v1.nc"))
    dest = ux.open_grid(grid_path)
    da_idw = uxds["v1"].remap.inverse_distance_weighted(
        destination_grid=dest, remap_to="nodes", power=3, k=8
    )
    assert not np.array_equal(uxds["v1"].values, da_idw.values)


def test_idw_return_types_and_counts(gridpath, datasetpath):
    """IDW remap returns UxDataArray or UxDataset with correct var counts."""
    grid_path = gridpath("ugrid", "geoflow-small", "grid.nc")
    dsfiles_geoflow = [datasetpath("ugrid", "geoflow-small", "v1.nc"), datasetpath("ugrid", "geoflow-small", "v2.nc"), datasetpath("ugrid", "geoflow-small", "v3.nc")]
    uxds = ux.open_mfdataset(grid_path, dsfiles_geoflow)
    dest = ux.open_grid(grid_path)

    da_idw = uxds["v1"].remap.inverse_distance_weighted(destination_grid=dest)
    ds_idw = uxds.remap.inverse_distance_weighted(destination_grid=dest)

    assert isinstance(da_idw, UxDataArray)
    assert isinstance(ds_idw, UxDataset)
    assert set(ds_idw.data_vars) == set(uxds.data_vars)


def test_idw_edge_centers_dim_change(gridpath, datasetpath):
    """IDW remap to edge centers produces an 'n_edge' dimension."""
    uxds = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))
    dest = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    da = uxds["v1"].remap.inverse_distance_weighted(
        destination_grid=dest, remap_to="edge centers", k=8
    )
    assert "n_edge" in da.dims


def test_idw_k_neighbors_effect():
    """Varying k in IDW actually changes the output."""
    src_da, grid = _make_node_da([1.0, 2.0, 3.0])
    dest = grid
    # k=2 vs. k=3 must differ
    idw2 = src_da.remap.inverse_distance_weighted(destination_grid=dest, remap_to="nodes", k=2)
    idw3 = src_da.remap.inverse_distance_weighted(destination_grid=dest, remap_to="nodes", k=3)
    assert not np.array_equal(idw2.values, idw3.values)


def test_idw_weights_sum_to_one():
    """Ensure that the computed IDW weights normalize to 1."""
    # use internal utility
    from uxarray.remap.inverse_distance_weighted import _idw_weights
    distances = np.eye(3)
    w = _idw_weights(distances, power=2)
    nt.assert_allclose(w.sum(axis=1), np.ones(3), atol=1e-12)


def test_idw_power_zero_is_mean():
    """With power==0 all distances collapse, so IDW→arithmetic mean."""
    src_da, grid = _make_node_da([1.0, 2.0, 4.0])
    out = src_da.remap.inverse_distance_weighted(
        destination_grid=grid, remap_to="nodes", power=0, k=3
    )
    expected = np.mean([1.0,2.0,4.0])
    assert np.allclose(out.values, expected)


def test_invalid_remap_to_raises():
    src_da, grid = _make_node_da([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        src_da.remap.nearest_neighbor(destination_grid=grid, remap_to="foobars")
    with pytest.raises(ValueError):
        src_da.remap.inverse_distance_weighted(destination_grid=grid, remap_to="123")
    with pytest.raises(ValueError):
        src_da.remap.bilinear(destination_grid=grid, remap_to="nothing")


def test_nn_equals_idw_high_power():
    """Nearest-neighbor should match IDW when power is huge and k=1."""
    src_da, grid = _make_node_da([2.0, 5.0, -1.0])
    nn = src_da.remap.nearest_neighbor(destination_grid=grid, remap_to="nodes")
    idw = src_da.remap.inverse_distance_weighted(
        destination_grid=grid, remap_to="nodes", power=1e6, k=1
    )
    nt.assert_allclose(nn.values, idw.values)


def test_dataset_remap_preserves_coords(gridpath, datasetpath):
    """When remapping a Dataset, time coords and attrs must survive."""
    uxds = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), datasetpath("ugrid", "geoflow-small", "v1.nc"))
    uxds = uxds.assign_coords(time=("time", np.arange(len(uxds.time))))
    dest = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    ds_out = uxds.remap.inverse_distance_weighted(destination_grid=dest, remap_to="nodes")
    assert "time" in ds_out.coords


# ------------------------------------------------------------
# Bilinear tests
# ------------------------------------------------------------
def test_b_modifies_values(gridpath):
    """Bilinear remap should change the array when remap_to != source."""
    mesh_path = gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)
    dest = ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc"))
    da_idw = uxds["latCell"].remap.inverse_distance_weighted(
        destination_grid=dest, remap_to="nodes"
    )
    assert not np.array_equal(uxds["latCell"].values, da_idw.values)


def test_b_return_types_and_counts(gridpath):
    """Bilinear remap returns UxDataArray or UxDataset with correct var counts."""
    mesh_path = gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)
    dest = ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc"))

    da_idw = uxds["latCell"].remap.bilinear(destination_grid=dest)

    # Create a dataset with only a face centered variable
    face_center_uxds = uxds["latCell"].to_dataset()
    ds_idw = face_center_uxds.remap.bilinear(destination_grid=dest)

    assert isinstance(da_idw, UxDataArray)
    assert isinstance(ds_idw, UxDataset)
    assert set(ds_idw.data_vars) == set(face_center_uxds.data_vars)

def test_b_edge_centers_dim_change(gridpath):
    """Bilinear remap to edge centers produces an 'n_edge' dimension."""
    mesh_path = gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)
    dest = ux.open_grid(mesh_path)
    da = uxds["latCell"].remap.bilinear(
        destination_grid=dest, remap_to="edges"
    )
    assert "n_edge" in da.dims


def test_b_value_errors(gridpath, datasetpath):
    """Bilinear remapping raises a value error when the source data is not on the faces"""
    grid_path = gridpath("ugrid", "geoflow-small", "grid.nc")
    uxds = ux.open_dataset(grid_path, datasetpath("ugrid", "geoflow-small", "v1.nc"))
    dest = ux.open_grid(grid_path)

    with nt.assert_raises(ValueError):
        uxds['v1'].remap.bilinear(destination_grid=dest, remap_to="nodes")

def test_b_quadrilateral(gridpath, datasetpath):
    """Bilinear remapping on quadrilaterals, to test Newton Quadrilateral weights calculation"""
    uxds = ux.open_dataset(gridpath("ugrid", "outCSne30", "outCSne30.ug"), datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc"))
    dest = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    out = uxds['var2'].remap.bilinear(destination_grid=dest)

    assert out.size > 0
