import uxarray as ux

import numpy as np
import numpy.testing as nt
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
    pytest.importorskip("dask")  # dask-backed branch requires dask
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))
    uxda = uxds['areaTriangle']

    for agg_func in AGGS:
        numpy_result = getattr(uxda, agg_func)(destination='face')
        dask_result = getattr(uxda.chunk(), agg_func)(destination='face')

        assert numpy_result.dims == dask_result.dims
        assert numpy_result.dtype == dask_result.dtype
        # both paths run the same kernel over the same partitions, so they must
        # agree exactly -- a mere allclose would hide a reordering regression
        nt.assert_array_equal(numpy_result.values, dask_result.values)


def test_node_to_edge_dask_reproduces_numpy(gridpath):
    # the numpy (eager) and dask (chunked) branches must agree
    pytest.importorskip("dask")  # dask-backed branch requires dask
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))
    uxda = uxds['areaTriangle']

    for agg_func in AGGS:
        numpy_result = getattr(uxda, agg_func)(destination='edge')
        dask_result = getattr(uxda.chunk(), agg_func)(destination='edge')

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
    pytest.importorskip("dask")  # dask-backed branch requires dask
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

        # three chunks along 'lev', so the kernel is applied three times
        assert len(dask_result.chunks[0]) == 3

        assert numpy_result.dims == dask_result.dims
        assert numpy_result.dtype == dask_result.dtype
        nt.assert_array_equal(numpy_result.values, dask_result.values)
