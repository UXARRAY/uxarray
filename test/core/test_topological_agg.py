
import dask.array as da
import uxarray as ux

import numpy as np
import numpy.testing as nt
import pandas as pd
import pytest




AGGS = ["topological_mean",
        "topological_max",
        "topological_min",
        "topological_prod",
        "topological_sum",
        "topological_std",
        "topological_std",
        "topological_var",
        "topological_median",
        "topological_all",
        "topological_any"]

def test_node_to_face_aggs(gridpath):
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))

    for agg_func in AGGS:
        grid_reduction = getattr(uxds['areaTriangle'], agg_func)(destination='face')

        assert 'n_face' in grid_reduction.dims

def test_node_to_edge_aggs(gridpath):
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))

    for agg_func in AGGS:
        grid_reduction = getattr(uxds['areaTriangle'], agg_func)(destination='edge')

        assert 'n_edge' in grid_reduction.dims


def test_node_to_face_dask_reproduces_numpy(gridpath):
    # the numpy (eager) and dask (chunked) branches must agree
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))
    uxda = uxds['areaTriangle']

    for agg_func in AGGS:
        numpy_result = getattr(uxda, agg_func)(destination='face')
        dask_result = getattr(uxda.chunk(), agg_func)(destination='face')

        assert isinstance(dask_result.data, da.Array)
        assert numpy_result.dims == dask_result.dims
        assert numpy_result.dtype == dask_result.dtype
        # both paths run the same kernel over the same partitions, so they must
        # agree exactly -- a mere allclose would hide a reordering regression
        nt.assert_array_equal(numpy_result.values, dask_result.values)


def test_node_to_edge_dask_reproduces_numpy(gridpath):
    # the numpy (eager) and dask (chunked) branches must agree
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))
    uxda = uxds['areaTriangle']

    for agg_func in AGGS:
        numpy_result = getattr(uxda, agg_func)(destination='edge')
        dask_result = getattr(uxda.chunk(), agg_func)(destination='edge')

        assert isinstance(dask_result.data, da.Array)
        assert numpy_result.dims == dask_result.dims
        assert numpy_result.dtype == dask_result.dtype
        # both paths run the same kernel over the same partitions, so they must
        # agree exactly -- a mere allclose would hide a reordering regression
        nt.assert_array_equal(numpy_result.values, dask_result.values)


@pytest.mark.parametrize("destination", ["face", "edge"])
def test_node_aggs_dask_reproduces_numpy_blockwise(gridpath, destination):
    # 'areaTriangle' is 1D, so chunking it leaves a single block and the
    # blockwise machinery is never exercised. Add a leading dimension so the
    # dask path really runs the kernel once per chunk.
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))
    uxgrid = uxds['areaTriangle'].uxgrid

    rng = np.random.default_rng(0)
    uxda = ux.UxDataArray(
        rng.random((6, uxgrid.n_node)), dims=("lev", "n_node"), uxgrid=uxgrid, name="var"
    )

    for agg_func in AGGS:
        numpy_result = getattr(uxda, agg_func)(destination=destination)
        chunked = uxda.chunk({"lev": 2})
        dask_result = getattr(chunked, agg_func)(destination=destination)

        assert isinstance(dask_result.data, da.Array)
        # three chunks along 'lev', so the kernel is applied three times
        assert len(dask_result.chunks[0]) == 3

        assert numpy_result.dims == dask_result.dims
        assert numpy_result.dtype == dask_result.dtype
        nt.assert_array_equal(numpy_result.values, dask_result.values)


def _timeseries_uxda(gridpath):
    """Node-centered data with a labelled time axis and CF-style attributes."""
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "oQU480.231010.nc"))
    rng = np.random.default_rng(0)
    return ux.UxDataArray(
        rng.random((6, uxgrid.n_node)),
        dims=("time", "n_node"),
        coords={"time": pd.date_range("2000-01-01", periods=6, freq="MS")},
        uxgrid=uxgrid,
        name="var",
        attrs={"units": "m", "long_name": "sea surface height"},
    )


@pytest.mark.parametrize("destination", ["face", "edge"])
def test_agg_preserves_leading_coords_and_attrs(gridpath, destination):
    """Aggregating over the node dimension must not discard the leading
    coordinates or the variable metadata. Regression test for topological
    aggregations returning a coordinate-less result, which broke label-based
    indexing (``.sel``/``.groupby``/``.resample``) on the output.
    """
    uxda = _timeseries_uxda(gridpath)

    for agg_func in AGGS:
        result = getattr(uxda, agg_func)(destination=destination)

        assert "time" in result.coords
        nt.assert_array_equal(result.time.values, uxda.time.values)
        assert result.attrs == uxda.attrs


@pytest.mark.parametrize("destination", ["face", "edge"])
def test_agg_result_supports_label_based_indexing(gridpath, destination):
    """The preserved time axis must actually be usable downstream."""
    result = _timeseries_uxda(gridpath).topological_mean(destination=destination)

    grid_dim = f"n_{destination}"
    assert result.sel(time="2000-03-01").dims == (grid_dim,)
    assert (
        result.groupby("time.season").mean().sizes[grid_dim] == result.sizes[grid_dim]
    )
    assert result.resample(time="QS").mean().sizes["time"] == 2


@pytest.mark.parametrize("destination", ["face", "edge"])
def test_agg_drops_node_spanning_coords(gridpath, destination):
    """Coordinates along the reduced dimension cannot be carried over, since
    they no longer match the length of the output dimension.
    """
    uxda = _timeseries_uxda(gridpath)
    rng = np.random.default_rng(1)
    uxda = uxda.assign_coords(node_lon=("n_node", rng.random(uxda.uxgrid.n_node)))

    result = uxda.topological_mean(destination=destination)

    assert "node_lon" not in result.coords
    assert "n_node" not in result.dims
    assert "time" in result.coords

