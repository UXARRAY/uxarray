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
