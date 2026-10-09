import numpy as np
import pytest
import xarray as xr

from uxarray.utils.coords import _preserve_valid_coords


@pytest.fixture
def da():
    """A DataArray carrying every kind of coordinate the helper must classify."""
    return xr.DataArray(
        np.zeros((2, 3)),
        dims=("time", "n_face"),
        coords={
            "time": [1, 2],
            "n_face": [0, 1, 2],
            "lat": ("n_face", [10.0, 20.0, 30.0]),
            "scalar": 5,
        },
    )


def test_drops_coords_spanning_dropped_dim(da):
    coords = _preserve_valid_coords(da, "n_face")

    assert set(coords) == {"time", "scalar"}


def test_keeps_everything_when_no_filters_given(da):
    coords = _preserve_valid_coords(da)

    assert set(coords) == set(da.coords)


def test_output_dims_drops_coords_on_absent_dims(da):
    """A coordinate on a dimension missing from the result cannot be carried."""
    coords = _preserve_valid_coords(da, output_dims={"time"})

    assert set(coords) == {"time", "scalar"}


def test_scalar_coords_always_survive(da):
    """Dimensionless coords span nothing, so no filter can invalidate them."""
    coords = _preserve_valid_coords(da, "n_face", output_dims=set())

    assert set(coords) == {"scalar"}


def test_exclude_drops_by_name_regardless_of_dims(da):
    coords = _preserve_valid_coords(da, "n_face", exclude={"scalar"})

    assert set(coords) == {"time"}


def test_dropped_dim_and_output_dims_compose(da):
    """Both filters apply; a coord must satisfy each one to survive."""
    coords = _preserve_valid_coords(da, "n_face", output_dims={"n_face", "n_lat"})

    assert set(coords) == {"scalar"}


def test_returns_the_original_coordinate_objects(da):
    coords = _preserve_valid_coords(da, "n_face")

    assert coords["time"].equals(da.coords["time"])


def test_result_is_accepted_by_the_dataarray_constructor(da):
    """The mapping must be usable directly as a ``coords`` argument."""
    coords = _preserve_valid_coords(da, "n_face")

    result = xr.DataArray(np.zeros((2, 4)), dims=("time", "n_edge"), coords=coords)

    assert result.sel(time=1).sizes == {"n_edge": 4}


def test_works_on_datasets(da):
    ds = da.to_dataset(name="v")

    coords = _preserve_valid_coords(ds, "n_face")

    assert set(coords) == {"time", "scalar"}
