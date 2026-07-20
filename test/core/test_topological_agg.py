import uxarray as ux

import numpy as np
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


def test_node_to_face_numpy_dask_allclose(gridpath):
    # the numpy (eager) and dask (chunked) branches must agree
    pytest.importorskip("dask")  # dask-backed branch requires dask
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))
    uxda = uxds['areaTriangle']

    for agg_func in AGGS:
        numpy_result = getattr(uxda, agg_func)(destination='face')
        dask_result = getattr(uxda.chunk(), agg_func)(destination='face')

        assert numpy_result.dims == dask_result.dims
        assert np.allclose(numpy_result.values, dask_result.values, equal_nan=True)


def test_node_to_edge_numpy_dask_allclose(gridpath):
    # the numpy (eager) and dask (chunked) branches must agree
    pytest.importorskip("dask")  # dask-backed branch requires dask
    uxds = ux.open_dataset(gridpath("mpas", "QU", "oQU480.231010.nc"), gridpath("mpas", "QU", "oQU480.231010.nc"))
    uxda = uxds['areaTriangle']

    for agg_func in AGGS:
        numpy_result = getattr(uxda, agg_func)(destination='edge')
        dask_result = getattr(uxda.chunk(), agg_func)(destination='edge')

        assert numpy_result.dims == dask_result.dims
        assert np.allclose(numpy_result.values, dask_result.values, equal_nan=True)
