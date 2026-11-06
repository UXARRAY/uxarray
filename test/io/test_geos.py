import uxarray as ux
import numpy as np
import pytest
import xarray as xr





def test_read_geos_cs_grid(gridpath):
    """Tests the conversion of a CS12 GEOS-CS Grid to the UGRID conventions.

    A CS12 grid has 6 faces, each with 12x12 faces and 13x13 nodes each.
    """
    uxgrid = ux.open_grid(gridpath("geos-cs", "c12", "test-c12.native.nc4"))

    n_face = 6 * 12 * 12
    n_node = 6 * 13 * 13

    assert uxgrid.n_face == n_face
    assert uxgrid.n_node == n_node

def test_read_geos_cs_uxds(gridpath):
    """Tests the creating of a UxDataset from a CS12 GEOS-CS Grid."""
    uxds = ux.open_dataset(gridpath("geos-cs", "c12", "test-c12.native.nc4"), gridpath("geos-cs", "c12", "test-c12.native.nc4"))

    assert uxds['T'].shape[-1] == uxds.uxgrid.n_face


@pytest.fixture
def six_tile_grid():
    """Six GEOS-CS tiles with one face and four nodes each."""
    lon = np.array(
        [[[x, x + 10], [x, x + 10]] for x in [-150, -90, -30, 30, 90, 150]],
        dtype=float,
    )
    lat = np.tile([[-10.0, -10.0], [10.0, 10.0]], (6, 1, 1))
    mesh = xr.Dataset(
        {
            "corner_lons": (("nf", "YCdim", "XCdim"), lon),
            "corner_lats": (("nf", "YCdim", "XCdim"), lat),
        }
    )
    return ux.Grid.from_dataset(mesh)


@pytest.mark.parametrize(
    "dims, ugrid_dim",
    [
        (("time", "nf", "Ydim", "Xdim"), "n_face"),
        (("time", "nf", "YCdim", "XCdim"), "n_node"),
    ],
)
def test_geos_cs_dataarray_from_xarray(six_tile_grid, dims, ugrid_dim):
    """UxDataArray.from_xarray flattens face and node data in the Grid's order."""
    n = 1 if ugrid_dim == "n_face" else 2
    da = xr.DataArray(
        np.arange(2 * 6 * n * n, dtype=float).reshape(2, 6, n, n),
        dims=dims,
        name="signal",
        attrs={"units": "K"},
    )

    uxda = ux.UxDataArray.from_xarray(da, uxgrid=six_tile_grid)

    assert uxda.dims == ("time", ugrid_dim)
    assert uxda.name == "signal"
    assert uxda.attrs == {"units": "K"}
    assert uxda.uxgrid is six_tile_grid
    np.testing.assert_array_equal(uxda.values, da.values.reshape(2, -1))

    # spatial dimension order does not change which value lands on which face
    transposed = da.transpose(dims[3], "time", dims[2], "nf")
    uxda_t = ux.UxDataArray.from_xarray(transposed, uxgrid=six_tile_grid)
    np.testing.assert_array_equal(uxda_t.transpose(*uxda.dims).values, uxda.values)


def test_geos_cs_dataarray_from_xarray_lazy(six_tile_grid):
    """Chunked input stays lazy."""
    da = xr.DataArray(
        np.arange(12.0).reshape(2, 6, 1, 1), dims=("time", "nf", "Ydim", "Xdim")
    ).chunk({"time": 1})

    uxda = ux.UxDataArray.from_xarray(da, uxgrid=six_tile_grid)

    assert uxda.chunks is not None
    np.testing.assert_array_equal(uxda.values, np.arange(12.0).reshape(2, 6))


def test_geos_cs_dataarray_matches_dataset(gridpath):
    """A variable converted on its own matches the same variable from open_dataset."""
    path = gridpath("geos-cs", "c12", "test-c12.native.nc4")
    uxds = ux.open_dataset(path, path)

    with xr.open_dataset(path) as xrds:
        uxda = ux.UxDataArray.from_xarray(xrds["T"], uxgrid=uxds.uxgrid)
        transposed = xrds["T"].transpose("Xdim", "lev", "nf", "time", "Ydim")
        uxda_t = ux.UxDataArray.from_xarray(transposed, uxgrid=uxds.uxgrid)

    xr.testing.assert_identical(uxda, uxds["T"])
    xr.testing.assert_identical(uxda_t.transpose(*uxda.dims), uxda)
    assert "n_face" not in uxda.indexes


def test_geos_cs_from_xarray_leaves_input_unchanged(gridpath):
    """UxDataset.from_xarray does not stack the caller's Dataset in place."""
    path = gridpath("geos-cs", "c12", "test-c12.native.nc4")
    uxgrid = ux.open_grid(path)

    with xr.open_dataset(path) as xrds:
        dims_before = xrds["T"].dims
        ux.UxDataset.from_xarray(xrds, uxgrid=uxgrid)

        assert xrds["T"].dims == dims_before
