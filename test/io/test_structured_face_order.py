"""Synthetic regressions for structured data-to-face alignment (GH #1828)."""

import numpy as np
import pytest
import xarray as xr

import uxarray as ux
from uxarray.errors import DimensionError


def _structured_dataset(reverse_variables=False, time_order=None, lazy=False):
    """Return equal fields with deliberately different spatial dimension orders."""
    ds = xr.Dataset(
        {
            "yfirst": (("y", "x"), [[0.0, 20.0, 40.0], [100.0, 120.0, 140.0]]),
            "xfirst": (("x", "y"), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        },
        coords={
            "x": ("x", [0.0, 20.0, 40.0], {"standard_name": "longitude"}),
            "y": ("y", [-10.0, 50.0], {"standard_name": "latitude"}),
        },
        attrs={"synthetic": "GH1828"},
    )
    if reverse_variables:
        ds = ds[["xfirst", "yfirst"]]
    if time_order is not None:
        ds = xr.concat([ds, ds + 1000.0], dim=xr.IndexVariable("time", [0, 1]))
        ds = ds.transpose(*time_order)
    if lazy:
        ds = ds.chunk({"x": 1, "y": 1})
    return ds


def _assert_face_alignment(mapped, original):
    """Check each value against geometry, not just another flattened dataset."""
    grid = mapped.uxgrid
    assert grid._source_dims_dict == {"n_face": ("x", "y")}
    # The six non-polar faces have four nodes. Read their locations to establish
    # the physical row-major face order independently of the mapping helper.
    nodes = grid.face_node_connectivity.values
    np.testing.assert_allclose(
        grid.node_lon.values[nodes].mean(axis=1), np.tile(original.x.values, 2)
    )
    np.testing.assert_allclose(
        grid.node_lat.values[nodes].mean(axis=1), np.repeat(original.y.values, 3)
    )
    for name in original.data_vars:
        other_dims = [dim for dim in original[name].dims if dim not in ("x", "y")]
        field = original[name].transpose(*other_dims, "y", "x")
        expected = field.values.reshape(*field.shape[:-2], 6)
        actual = mapped[name].transpose(*other_dims, "n_face")
        np.testing.assert_array_equal(actual.values, expected)
        np.testing.assert_allclose(
            actual.integrate().values,
            np.sum(expected * grid.face_areas.values, axis=-1),
        )


@pytest.mark.parametrize("constructor", ["from_xarray", "from_structured"])
@pytest.mark.parametrize("reverse_variables", [False, True])
@pytest.mark.parametrize("time_order", [None, ("time", "y", "x"), ("x", "time", "y")])
@pytest.mark.parametrize("lazy", [False, True])
def test_structured_face_order(constructor, reverse_variables, time_order, lazy):
    """Variable insertion order, axis order, and chunks cannot change face values."""
    ds = _structured_dataset(reverse_variables, time_order, lazy)
    original = ds.copy(deep=True)
    grid = ux.Grid.from_structured(lon=ds.x, lat=ds.y)
    if constructor == "from_xarray":
        mapped = ux.UxDataset.from_xarray(ds, uxgrid=grid)
        assert mapped.uxgrid is grid
    else:
        mapped = ux.UxDataset.from_structured(ds)
    if lazy:
        assert all(mapped[name].chunks is not None for name in ds.data_vars)
    xr.testing.assert_identical(ds, original)
    _assert_face_alignment(mapped, original)


def test_structured_variable_order_preserves_face_index():
    """Equal datasets must retain equal n_face indexes after conversion."""
    ds0 = _structured_dataset()
    ds1 = _structured_dataset(reverse_variables=True)
    assert ds0.equals(ds1)
    grid = ux.Grid.from_structured(lon=ds0.x, lat=ds0.y)
    mapped0 = ux.UxDataset.from_xarray(ds0, uxgrid=grid)
    mapped1 = ux.UxDataset.from_xarray(ds1, uxgrid=grid)
    assert mapped0.indexes["n_face"].equals(mapped1.indexes["n_face"])
    for name in ds0.data_vars:
        xr.testing.assert_identical(mapped0[name], mapped1[name])


@pytest.mark.parametrize("square", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_structured_explicit_mapping_with_renamed_dimensions(square, transpose):
    """Explicit mappings may use names unlike the Grid's source dimensions."""
    lon = xr.DataArray([0.0, 20.0, 40.0], dims="x")
    lat = xr.DataArray([-10.0, 20.0, 50.0] if square else [-10.0, 50.0], dims="y")
    grid = ux.Grid.from_structured(lon=lon, lat=lat)
    ds = xr.Dataset({"field": 2.0 * lat + lon}).rename({"x": "column", "y": "row"})
    if transpose:
        ds = ds.transpose("column", "row")
    original = ds.copy(deep=True)
    grid_data = grid._ds.copy(deep=True)
    source_dims = grid._source_dims_dict.copy()
    mapping = {"n_face": ("column", "row")}

    mapped = ux.UxDataset.from_xarray(ds, uxgrid=grid, ugrid_dims=mapping)

    assert mapped.uxgrid is grid
    assert grid.source_grid_spec == "Structured"
    assert grid._source_dims_dict == source_dims == {"n_face": ("x", "y")}
    assert mapping == {"n_face": ("column", "row")}
    xr.testing.assert_identical(grid._ds, grid_data)
    xr.testing.assert_identical(ds, original)
    # Recover each face's physical center independently of dimension stacking.
    nodes = grid.face_node_connectivity.values
    face_lon = grid.node_lon.values[nodes].mean(axis=1)
    face_lat = grid.node_lat.values[nodes].mean(axis=1)
    assert mapped.field.dims == ("n_face",)
    np.testing.assert_allclose(mapped.field.values, 2.0 * face_lat + face_lon)


@pytest.mark.parametrize("transpose", [False, True])
def test_structured_netcdf_face_selection(tmp_path, transpose):
    """Read a member file, attach the same Grid, and select the correct face."""
    ds = _structured_dataset()
    if transpose:
        ds = ds.transpose("x", "y")
    grid = ux.Grid.from_structured(lon=ds.x, lat=ds.y)
    path = tmp_path / "synthetic-member.nc"
    ds.to_netcdf(path)
    with xr.open_dataset(path) as reopened:
        mapped = ux.UxDataset.from_xarray(reopened, uxgrid=grid)
        _assert_face_alignment(mapped, reopened)
        selected = mapped["yfirst"].isel(n_face=[1, 4])
        np.testing.assert_array_equal(selected.values, [20.0, 120.0])
        assert selected.uxgrid.n_face == 2


@pytest.mark.parametrize("missing", ["x", "y", "both"])
def test_structured_missing_source_dimension_raises(missing):
    """Do not silently produce an unflattened result for an incompatible Grid."""
    ds = _structured_dataset()
    grid = ux.Grid.from_structured(lon=ds.x, lat=ds.y)
    ds = ds.isel({dim: 0 for dim in ("x", "y") if missing in (dim, "both")})
    with pytest.raises(DimensionError, match="selected mapping's spatial dimensions"):
        ux.UxDataset.from_xarray(ds, uxgrid=grid)


def test_structured_coordinate_name_mapping_mismatch_raises():
    """Keep bad source metadata visible instead of silently retaining x/y."""
    ds = _structured_dataset().rename_vars({"x": "longitude", "y": "latitude"})
    # Grid.from_dataset currently records coordinate names rather than dimension
    # names for this input. Correcting that metadata is separate from GH1828.
    with pytest.raises(DimensionError, match="selected mapping's spatial dimensions"):
        ux.UxDataset.from_structured(ds)
