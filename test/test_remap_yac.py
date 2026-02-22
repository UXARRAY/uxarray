import numpy as np
import pytest

import uxarray as ux


yac = pytest.importorskip("yac")


def test_yac_nnn_node_remap(gridpath, datasetpath):
    grid_path = gridpath("ugrid", "geoflow-small", "grid.nc")
    uxds = ux.open_dataset(grid_path, datasetpath("ugrid", "geoflow-small", "v1.nc"))
    dest = ux.open_grid(grid_path)

    out = uxds["v1"].remap.nearest_neighbor(
        destination_grid=dest,
        remap_to="nodes",
        backend="yac",
        yac_method="nnn",
        yac_options={"n": 1},
    )
    assert out.size > 0
    assert "n_node" in out.dims


def test_yac_conservative_face_remap(gridpath):
    mesh_path = gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)
    dest = ux.open_grid(mesh_path)

    out = uxds["latCell"].remap.nearest_neighbor(
        destination_grid=dest,
        remap_to="faces",
        backend="yac",
        yac_method="conservative",
        yac_options={"order": 1},
    )
    assert out.size == dest.n_face


def test_yac_matches_uxarray_nearest_neighbor():
    verts = np.array([(0.0, 90.0), (-180.0, 0.0), (0.0, -90.0)])
    grid = ux.open_grid(verts)
    da = ux.UxDataArray(
        np.asarray([1.0, 2.0, 3.0]),
        dims=["n_node"],
        coords={"n_node": [0, 1, 2]},
        uxgrid=grid,
    )

    ux_out = da.remap.nearest_neighbor(
        destination_grid=grid,
        remap_to="nodes",
        backend="uxarray",
    )
    yac_out = da.remap.nearest_neighbor(
        destination_grid=grid,
        remap_to="nodes",
        backend="yac",
        yac_method="nnn",
        yac_options={"n": 1},
    )
    assert ux_out.shape == yac_out.shape
    assert (ux_out.values == yac_out.values).all()
