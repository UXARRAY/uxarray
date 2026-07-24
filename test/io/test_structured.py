import uxarray as ux
import xarray as xr
import pytest


@pytest.mark.parametrize("ds_name", ["air_temperature", "ersstv5"])
def test_read_structured_grid_from_ds(ds_name):
    ds = xr.tutorial.open_dataset(ds_name)
    uxgrid = ux.Grid.from_structured(ds)

    assert uxgrid.n_face == ds.sizes['lon'] * ds.sizes['lat']

    assert uxgrid.validate()


@pytest.mark.parametrize("ds_name", ["air_temperature", "ersstv5"])
def test_read_structured_grid_from_latlon(ds_name):
    ds = xr.tutorial.open_dataset(ds_name)
    uxgrid = ux.Grid.from_structured(lon=ds.lon, lat=ds.lat)
    assert uxgrid.n_face == ds.sizes['lon'] * ds.sizes['lat']
    assert uxgrid.validate()

@pytest.mark.parametrize("ds_name", ["air_temperature", "ersstv5"])
def test_read_structured_uxds_from_ds(ds_name):
    # Load the dataset using xarray's tutorial module
    ds = xr.tutorial.open_dataset(ds_name)

    # Create a uxarray Grid from the structured dataset
    uxds = ux.UxDataset.from_structured(ds)

    assert "n_face" in uxds.dims

    assert "lon" not in uxds.dims
    assert "lat" not in uxds.dims

    assert uxds.uxgrid.validate()


@pytest.mark.parametrize("ds_name", ["air_temperature"])
def test_from_xarray_with_grid_from_latlon(ds_name):
    """Regression test for GH #1410: a Grid built via ``from_structured(lon=, lat=)``
    must record its source dimensions so ``UxDataset.from_xarray`` flattens the
    structured (lon, lat) data variables onto ``n_face``."""
    ds = xr.tutorial.open_dataset(ds_name)

    uxgrid = ux.Grid.from_structured(lon=ds.lon, lat=ds.lat)

    # The grid must carry the structured spec and the source-dim mapping.
    assert uxgrid.source_grid_spec == "Structured"
    assert uxgrid._source_dims_dict == {"n_face": (ds.lon.dims[0], ds.lat.dims[0])}

    uxds = ux.UxDataset.from_xarray(ds, uxgrid=uxgrid)

    # Data must now be mapped onto n_face, not left on lon/lat.
    assert "n_face" in uxds.dims
    assert "lon" not in uxds.dims
    assert "lat" not in uxds.dims
    assert uxds["air"].sizes["n_face"] == ds.sizes["lon"] * ds.sizes["lat"]

    # The flatten must preserve data order: each face value must equal the
    # original (lat, lon) cell value at that face. n_face is stacked as
    # (lat, lon) C-order, so face k corresponds to (k // n_lon, k % n_lon).
    n_lon = ds.sizes["lon"]
    original = ds["air"].isel(time=0).values  # (lat, lon)
    flattened = uxds["air"].isel(time=0).values  # (n_face,)
    for k in (0, n_lon + 1, flattened.size - 1):  # first, an interior, last
        i, j = k // n_lon, k % n_lon
        assert flattened[k] == original[i, j]

    # End-to-end: the mapped data must be subsettable (the symptom in #1410).
    subset = uxds["air"].isel(time=0).subset.bounding_circle((-100.0, 40.0), 5)
    assert "n_face" in subset.dims
    assert subset.sizes["n_face"] > 0
