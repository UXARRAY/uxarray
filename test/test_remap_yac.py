import numpy as np
import pytest

import uxarray as ux
from uxarray.remap.yac import YacNotAvailableError, _import_yac


try:
    _import_yac()
except YacNotAvailableError:
    pytest.skip("yac.core is not available", allow_module_level=True)


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

    out = uxds["latCell"].remap(
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


def test_yac_call_defaults_to_nnn():
    verts = np.array([(0.0, 90.0), (-180.0, 0.0), (0.0, -90.0)])
    grid = ux.open_grid(verts)
    da = ux.UxDataArray(
        np.asarray([1.0, 2.0, 3.0]),
        dims=["n_node"],
        coords={"n_node": [0, 1, 2]},
        uxgrid=grid,
    )

    out = da.remap(
        destination_grid=grid,
        remap_to="nodes",
        backend="yac",
    )

    assert out.shape == da.shape
    np.testing.assert_array_equal(out.values, da.values)


def test_yac_invalid_backend_raises():
    verts = np.array([(0.0, 90.0), (-180.0, 0.0), (0.0, -90.0)])
    grid = ux.open_grid(verts)
    da = ux.UxDataArray(
        np.asarray([1.0, 2.0, 3.0]),
        dims=["n_node"],
        coords={"n_node": [0, 1, 2]},
        uxgrid=grid,
    )

    with pytest.raises(ValueError, match="Invalid backend"):
        da.remap.nearest_neighbor(
            destination_grid=grid,
            remap_to="nodes",
            backend="bogus",
        )


def test_yac_idw_not_implemented():
    verts = np.array([(0.0, 90.0), (-180.0, 0.0), (0.0, -90.0)])
    grid = ux.open_grid(verts)
    da = ux.UxDataArray(
        np.asarray([1.0, 2.0, 3.0]),
        dims=["n_node"],
        coords={"n_node": [0, 1, 2]},
        uxgrid=grid,
    )

    with pytest.raises(NotImplementedError, match="inverse_distance_weighted"):
        da.remap.inverse_distance_weighted(
            destination_grid=grid,
            remap_to="nodes",
            backend="yac",
            yac_method="nnn",
            yac_options={"n": 1},
        )


def test_yac_bilinear_face_remap(gridpath):
    mesh_path = gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)
    dest = ux.open_grid(mesh_path)

    out = uxds["latCell"].remap.bilinear(
        destination_grid=dest,
        remap_to="faces",
        backend="yac",
    )

    assert out.size == dest.n_face


def test_yac_bilinear_rejects_non_average_method(gridpath):
    mesh_path = gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc")
    uxds = ux.open_dataset(mesh_path, mesh_path)
    dest = ux.open_grid(mesh_path)

    with pytest.raises(ValueError, match="only supports yac_method='average'"):
        uxds["latCell"].remap.bilinear(
            destination_grid=dest,
            remap_to="faces",
            backend="yac",
            yac_method="conservative",
        )


def test_yac_conservative_rejects_non_face_data():
    verts = np.array([(0.0, 90.0), (-180.0, 0.0), (0.0, -90.0)])
    grid = ux.open_grid(verts)
    da = ux.UxDataArray(
        np.asarray([1.0, 2.0, 3.0]),
        dims=["n_node"],
        coords={"n_node": [0, 1, 2]},
        uxgrid=grid,
    )

    with pytest.raises(ValueError, match="face-centered"):
        da.remap.nearest_neighbor(
            destination_grid=grid,
            remap_to="nodes",
            backend="yac",
            yac_method="conservative",
            yac_options={"order": 1},
        )


def test_yac_preserves_spatial_coordinate_remap():
    verts = np.array([(0.0, 90.0), (-180.0, 0.0), (0.0, -90.0)])
    grid = ux.open_grid(verts)
    da = ux.UxDataArray(
        np.asarray([1.0, 2.0, 3.0]),
        dims=["n_node"],
        coords={
            "n_node": [0, 1, 2],
            "node_lon": (
                "n_node",
                np.array([0.0, -180.0, 0.0]),
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "node_lat": (
                "n_node",
                np.array([90.0, 0.0, -90.0]),
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
        },
        uxgrid=grid,
    )

    out = da.remap.nearest_neighbor(
        destination_grid=grid,
        remap_to="nodes",
        backend="yac",
        yac_method="nnn",
        yac_options={"n": 1},
    )

    np.testing.assert_array_equal(out.values, da.values)
    assert "node_lon" in out.coords
    assert "node_lat" in out.coords


def test_yac_batched_remap_with_extra_dimension():
    verts = np.array([(0.0, 90.0), (-180.0, 0.0), (0.0, -90.0)])
    grid = ux.open_grid(verts)
    da = ux.UxDataArray(
        np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
        dims=["time", "n_node"],
        coords={"time": [0, 1], "n_node": [0, 1, 2]},
        uxgrid=grid,
    )

    out = da.remap.nearest_neighbor(
        destination_grid=grid,
        remap_to="nodes",
        backend="yac",
        yac_method="nnn",
        yac_options={"n": 1},
    )

    assert out.shape == da.shape
    np.testing.assert_array_equal(out.values, da.values)


def test_yac_batched_remap_with_fractional_mask():
    verts = np.array([(0.0, 90.0), (-180.0, 0.0), (0.0, -90.0)])
    grid = ux.open_grid(verts)
    da = ux.UxDataArray(
        np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
        dims=["time", "n_node"],
        coords={"time": [0, 1], "n_node": [0, 1, 2]},
        uxgrid=grid,
    )
    frac_mask = np.ones_like(da.values, dtype=np.float64)

    out = da.remap.nearest_neighbor(
        destination_grid=grid,
        remap_to="nodes",
        backend="yac",
        yac_method="nnn",
        yac_options={
            "n": 1,
            "frac_mask_fallback_value": 0.0,
            "frac_mask": frac_mask,
        },
    )

    assert out.shape == da.shape
    np.testing.assert_array_equal(out.values, da.values)


def test_yac_to_rectilinear_node_remap():
    verts = np.array([(0.0, 0.0), (90.0, 0.0), (0.0, 45.0)])
    grid = ux.open_grid(verts)
    da = ux.UxDataArray(
        np.asarray([1.0, 2.0, 3.0]),
        dims=["n_node"],
        coords={"n_node": [0, 1, 2]},
        uxgrid=grid,
    )

    out = da.remap.to_rectilinear(
        lon=np.asarray([0.0, 90.0]),
        lat=np.asarray([0.0, 45.0]),
        backend="yac",
        yac_method="nnn",
        yac_options={"n": 1},
    )

    assert out.dims == ("lat", "lon")
    assert out.shape == (2, 2)
    np.testing.assert_array_equal(out.values, np.asarray([[1.0, 3.0], [2.0, 3.0]]))


def test_yac_to_rectilinear_preserves_extra_dimensions():
    verts = np.array([(0.0, 0.0), (90.0, 0.0), (0.0, 45.0)])
    grid = ux.open_grid(verts)
    da = ux.UxDataArray(
        np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
        dims=["time", "n_node"],
        coords={"time": [0, 1], "n_node": [0, 1, 2]},
        uxgrid=grid,
    )

    out = da.remap.to_rectilinear(
        lon=np.asarray([0.0, 90.0]),
        lat=np.asarray([0.0, 45.0]),
        backend="yac",
        yac_method="nnn",
        yac_options={"n": 1},
    )

    assert out.dims == ("time", "lat", "lon")
    assert out.shape == (2, 2, 2)
    np.testing.assert_array_equal(
        out.values,
        np.asarray([[[1.0, 3.0], [2.0, 3.0]], [[10.0, 30.0], [20.0, 30.0]]]),
    )
