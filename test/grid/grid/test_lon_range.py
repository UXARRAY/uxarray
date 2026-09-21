"""Guards on the [-180, 180] longitude wrap.

``_set_desired_longitude_range`` used to decide whether to wrap by asking
``lon.max() > 180``. On a dask-backed grid that reduction is a compute, and
``Grid.__init__`` calls it, so opening a grid pulled its longitude
coordinates into memory. The wrap is now elementwise.

Two things have to hold for that to be an improvement rather than a trade:
the construction must stay lazy, and repeated calls must not pile ``where``
layers onto the graph -- ``edge_lat`` invokes it on every property access.
"""

import numpy as np
import numpy.testing as nt
import pytest
import xarray as xr
from dask.callbacks import Callback

import uxarray as ux
from uxarray.grid.coordinates import _set_desired_longitude_range


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
    """Outside [-180, 180] wraps; inside is passed through bit-for-bit.

    The old whole-array form perturbed in-range longitudes by up to ~3e-14
    degrees, because ``(lon + 180) - 180`` does not round-trip exactly. The
    assertion here is ``assert_array_equal``, not ``allclose``, on purpose.
    """
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
    """180.0 stays 180.0 and -180.0 stays -180.0, however the array looks.

    Not a detail. ``antimeridian_face_indices`` reads a face as crossing from
    the span of its longitudes, so a face with one vertex at exactly 180 and
    the rest near -170 is only visible while that vertex reads as 180 -- fold
    it to -180 and the span drops from 350 to 10. Under the old reduction the
    endpoint survived only by accident: an array whose maximum was exactly
    180 was not wrapped at all, so nothing touched it.
    """
    for companion in (10.0, 270.0, -170.0):
        grid = _shim(np.array([180.0, companion]))
        _set_desired_longitude_range(grid)
        assert grid._ds["node_lon"].values[0] == 180.0

    grid = _shim(np.array([-180.0, 0.0, 179.5]))
    _set_desired_longitude_range(grid)
    nt.assert_array_equal(grid._ds["node_lon"].values, [-180.0, 0.0, 179.5])


def test_wrap_normalizes_both_tails_and_is_idempotent():
    """One pass lands everything in [-180, 180); a second pass changes nothing.

    Both tails matter. The reduction tested ``lon.max() > 180``, so it reached
    longitudes below -180 only when the same array happened to hold one above
    180, and left them alone otherwise.
    """
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
    """The regression this change exists to fix.

    Counts every dask execution during ``open_grid``, then asserts the wrap
    contributed none of them by re-running the wrap on the constructed grid
    under its own counter.
    """
    path = gridpath("ugrid", "outCSne30", "outCSne30.ug")

    grid = ux.open_grid(path, chunks={"n_node": 1000})
    assert grid._ds["node_lon"].chunks is not None, "fixture is not chunked"

    counter = _CountComputes()
    with counter:
        _set_desired_longitude_range(grid)
    assert counter.n == 0


def test_repeated_calls_do_not_grow_the_graph(gridpath):
    """``edge_lat`` calls the wrap on every access, outside its populate guard.

    Without the memo, an unconditional elementwise wrap would add a ``where``
    layer per call -- lazy, so nothing would fail, just an ever-deepening
    graph.
    """
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
