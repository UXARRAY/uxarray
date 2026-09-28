"""Tests for the elementwise longitude wrap in ``_set_desired_longitude_range``."""

from types import SimpleNamespace

import dask
import numpy as np
import numpy.testing as nt
import pytest
import xarray as xr

import uxarray as ux
from uxarray.grid.coordinates import _set_desired_longitude_range

# Longitudes from outCSne30 that the old whole-array wrap perturbed.
TINY, NEAR_180 = 2.1182935802518936e-14, 179.99999999999997


@pytest.mark.parametrize(
    "values, expected",
    [
        ([TINY, NEAR_180, -45.0], [TINY, NEAR_180, -45.0]),
        ([TINY, NEAR_180, 270.0], [TINY, NEAR_180, -90.0]),
        ([180.0, -180.0, 270.0], [180.0, -180.0, -90.0]),
        ([-270.0, -540.5, 10.0], [90.0, 179.5, 10.0]),
        ([np.nan, 200.0], [np.nan, -160.0]),
    ],
    ids=["in-range", "mixed", "endpoints", "below-minus-180", "nan"],
)
def test_wrap(values, expected):
    """Out-of-range values wrap; in-range values, both endpoints included, pass
    through bit-for-bit."""
    lon = ("n_node", np.array(values), {"units": "degrees_east"})
    grid = SimpleNamespace(_ds=xr.Dataset({"node_lon": lon}))

    _set_desired_longitude_range(grid)

    out = grid._ds["node_lon"]
    nt.assert_array_equal(out.values, expected)
    assert out.name == "node_lon"
    assert out.attrs == {"units": "degrees_east"}


def _raise_on_compute(*args, **kwargs):
    raise AssertionError("_set_desired_longitude_range ran a dask compute")


def test_wrap_is_lazy_and_memoized(gridpath):
    """The wrap runs no dask compute, repeat calls add nothing to the graph, and
    a replaced coordinate is wrapped again."""
    grid = ux.open_grid(
        gridpath("ugrid", "outCSne30", "outCSne30.ug"), chunks={"n_node": 1000}
    )
    assert grid._ds["node_lon"].chunks is not None
    grid._wrapped_lon_vars = {}  # force a real wrap rather than a memo hit

    with dask.config.set(scheduler=_raise_on_compute):
        _set_desired_longitude_range(grid)
        n_tasks = len(grid._ds["node_lon"].data.dask)
        _set_desired_longitude_range(grid)
        _set_desired_longitude_range(grid)
    assert len(grid._ds["node_lon"].data.dask) == n_tasks

    grid._ds["node_lon"] = grid._ds["node_lon"] + 200.0
    _set_desired_longitude_range(grid)
    assert float(grid._ds["node_lon"].max()) <= 180.0
