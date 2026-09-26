"""Tests for the elementwise longitude wrap in ``_set_desired_longitude_range``.

It must stay lazy on dask-backed grids and must not grow the graph when called
repeatedly.
"""

import numpy as np
import numpy.testing as nt
import pytest
import xarray as xr
from dask.callbacks import Callback

import uxarray as ux
from uxarray.grid.coordinates import (
    _lon_within_range,
    _set_desired_longitude_range,
)


class _CountComputes(Callback):
    """Counts dask graph executions inside the ``with`` block."""

    def __init__(self):
        self.n = 0

    def _start(self, dsk):
        self.n += 1


class _Shim:
    """The only attribute ``_set_desired_longitude_range`` needs of a Grid."""

    def __init__(self, ds):
        self._ds = ds


def _shim(values, name="node_lon", **attrs):
    return _Shim(xr.Dataset({name: (f"n_{name.split('_')[0]}", values, attrs)}))


def test_wrap_is_elementwise_and_leaves_in_range_values_exact():
    """Outside [-180, 180] wraps; inside passes through bit-for-bit."""
    rng = np.random.default_rng(0)
    values = np.concatenate(
        [rng.uniform(0.0, 180.0, 500), rng.uniform(180.0, 360.0, 500)]
    )

    grid = _shim(values.copy())
    _set_desired_longitude_range(grid)
    wrapped = grid._ds["node_lon"].values

    in_range = values <= 180.0
    nt.assert_array_equal(wrapped[in_range], values[in_range])
    nt.assert_array_equal(
        wrapped[~in_range], (values[~in_range] + 180.0) % 360.0 - 180.0
    )
    assert wrapped.max() <= 180.0 and wrapped.min() >= -180.0


def test_both_endpoints_of_the_antimeridian_are_left_alone():
    """180.0 and -180.0 are both kept, whatever else the array holds.

    ``antimeridian_face_indices`` detects a crossing from a face's longitude span,
    which folding 180 to -180 would collapse.
    """
    for companion in (10.0, 270.0, -170.0):
        grid = _shim(np.array([180.0, companion]))
        _set_desired_longitude_range(grid)
        assert grid._ds["node_lon"].values[0] == 180.0

    grid = _shim(np.array([-180.0, 0.0, 179.5]))
    _set_desired_longitude_range(grid)
    nt.assert_array_equal(grid._ds["node_lon"].values, [-180.0, 0.0, 179.5])


def test_wrap_normalizes_both_tails_and_is_idempotent():
    """One pass lands everything in [-180, 180], below -180 too; a second is a no-op."""
    rng = np.random.default_rng(1)
    grid = _shim(rng.uniform(-720.0, 720.0, 1000))

    _set_desired_longitude_range(grid)
    once = grid._ds["node_lon"].values.copy()
    assert once.min() >= -180.0 and once.max() <= 180.0

    grid._wrapped_lon_vars = {}
    _set_desired_longitude_range(grid)
    nt.assert_array_equal(grid._ds["node_lon"].values, once)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_wrap_preserves_dtype_name_and_attrs(dtype):
    grid = _shim(np.array([0.0, 180.0, 270.0], dtype=dtype), units="degrees_east")
    _set_desired_longitude_range(grid)
    out = grid._ds["node_lon"]

    assert out.dtype == dtype
    assert out.name == "node_lon"
    assert out.attrs == {"units": "degrees_east"}


def test_nan_longitudes_survive_the_wrap():
    """``nan > 180`` is False, so NaN takes the pass-through branch."""
    grid = _shim(np.array([np.nan, 270.0, 10.0]))
    _set_desired_longitude_range(grid)

    nt.assert_array_equal(grid._ds["node_lon"].values, [np.nan, -90.0, 10.0])


def test_open_grid_does_not_compute_longitudes(gridpath):
    """The wrap runs no dask compute on a chunked grid, the regression fixed here."""
    path = gridpath("ugrid", "outCSne30", "outCSne30.ug")

    grid = ux.open_grid(path, chunks={"n_node": 1000})
    assert grid._ds["node_lon"].chunks is not None, "fixture is not chunked"

    counter = _CountComputes()
    with counter:
        _set_desired_longitude_range(grid)
    assert counter.n == 0


def test_repeated_calls_do_not_grow_the_graph(gridpath):
    """Repeated calls, as ``edge_lat`` makes, add no ``where`` layers to the graph."""
    path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
    grid = ux.open_grid(path, chunks={"n_node": 1000})

    before = len(grid._ds["node_lon"].data.dask)
    for _ in range(20):
        _set_desired_longitude_range(grid)
    assert len(grid._ds["node_lon"].data.dask) == before


def test_memo_reopens_when_the_variable_is_replaced(gridpath):
    """The memo keys on the ``xr.Variable``, so assignment invalidates it."""
    path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
    grid = ux.open_grid(path)

    grid._ds["node_lon"] = grid._ds["node_lon"] + 200.0
    assert float(grid._ds["node_lon"].max()) > 180.0

    _set_desired_longitude_range(grid)
    assert float(grid._ds["node_lon"].max()) <= 180.0


def test_eager_fast_path_agrees_with_the_where_element_for_element():
    """The in-memory guard must give exactly what the unguarded ``where`` gives."""
    rng = np.random.default_rng(2)
    cases = {
        "strictly inside": rng.uniform(-179.0, 179.0, 500),
        "on both endpoints": np.array([-180.0, 180.0, 0.0, 179.5, -179.5]),
        "needs wrapping": rng.uniform(0.0, 360.0, 500),
        "negative tail": rng.uniform(-540.0, -180.5, 500),
        "with nan": np.array([np.nan, 10.0, 200.0]),
    }

    for label, values in cases.items():
        da = xr.DataArray(values, dims="n_node", name="node_lon")
        unguarded = xr.where(
            (da > 180) | (da < -180), (da + 180) % 360 - 180, da
        ).values

        grid = _shim(values.copy())
        _set_desired_longitude_range(grid)

        nt.assert_array_equal(grid._ds["node_lon"].values, unguarded, err_msg=label)


def test_guard_is_skipped_for_dask_backed_arrays(gridpath):
    """``_lon_within_range`` computes on dask, so the wrap only runs it eagerly."""
    grid = ux.open_grid(
        gridpath("ugrid", "outCSne30", "outCSne30.ug"), chunks={"n_node": 1000}
    )
    da = grid._ds["node_lon"]
    assert da.chunks is not None

    counter = _CountComputes()
    with counter:
        _lon_within_range(da)
    assert counter.n > 0, "guard is lazy here; skipping it would be pointless"
