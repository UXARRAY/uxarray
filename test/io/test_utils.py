import numpy as np
import pytest
import xarray as xr

from uxarray.io.utils import _parse_grid_type


@pytest.mark.parametrize(
    ("path_args", "expected_spec"),
    [
        (("exodus", "outCSne8", "outCSne8.g"), "Exodus"),
        (("scrip", "outCSne8", "outCSne8.nc"), "Scrip"),
        (("ugrid", "outCSne30", "outCSne30.ug"), "UGRID"),
        (("mpas", "QU", "mesh.QU.1920km.151026.nc"), "MPAS"),
        (("esmf", "ne30", "ne30pg3.grid.nc"), "ESMF"),
        (("geos-cs", "c12", "test-c12.native.nc4"), "GEOS-CS"),
        (("icon", "R02B04", "icon_grid_0010_R02B04_G.nc"), "ICON"),
        (("fesom", "soufflet-netcdf", "grid.nc"), "FESOM2"),
    ],
)
def test_parse_grid_type_detects_supported_formats(gridpath, path_args, expected_spec):
    with xr.open_dataset(gridpath(*path_args)) as ds:
        source_grid_spec, lon_name, lat_name = _parse_grid_type(ds)

    assert source_grid_spec == expected_spec
    assert lon_name is None
    assert lat_name is None


def test_parse_grid_type_detects_structured_grid():
    lon = xr.DataArray(
        np.array([0.0, 1.0, 2.0]),
        dims=["lon"],
        attrs={"standard_name": "longitude"},
    )
    lat = xr.DataArray(
        np.array([-1.0, 0.0, 1.0]),
        dims=["lat"],
        attrs={"standard_name": "latitude"},
    )
    ds = xr.Dataset(coords={"lon": lon, "lat": lat})

    source_grid_spec, lon_name, lat_name = _parse_grid_type(ds)

    assert source_grid_spec == "Structured"
    assert lon_name == "lon"
    assert lat_name == "lat"


@pytest.mark.parametrize(
    "dataset",
    [
        xr.Dataset({"grid_center_lon": xr.DataArray([0.0], dims=["grid_size"])}),
        xr.Dataset(
            {
                "coordx": xr.DataArray([0.0, 1.0], dims=["num_nodes"]),
                "coordy": xr.DataArray([0.0, 1.0], dims=["num_nodes"]),
            }
        ),
        xr.Dataset({"verticesOnCell": xr.DataArray([[1, 2, 3]], dims=["nCells", "nVert"])}),
    ],
)
def test_parse_grid_type_rejects_incomplete_format_signals(dataset):
    with pytest.raises(RuntimeError, match="Failed to parse uxgrid information from xarray.Dataset."):
        _parse_grid_type(dataset)
