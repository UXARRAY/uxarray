
import dask.array as da
import numpy as np
import numpy.testing as nt
import pytest
import xarray as xr

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE


def test_single_dim(gridpath):
    """Integral with 1D data mapped to each face."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    test_data = np.ones(uxgrid.n_face)
    dims = {"n_face": uxgrid.n_face}
    uxda = ux.UxDataArray(data=test_data, dims=dims, uxgrid=uxgrid, name='var2')
    integral = uxda.integrate()
    assert integral.ndim == len(dims) - 1
    nt.assert_almost_equal(integral, 4 * np.pi)


def test_multi_dim(gridpath):
    """Integral with 3D data mapped to each face."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    test_data = np.ones((5, 5, uxgrid.n_face))
    dims = {"a": 5, "b": 5, "n_face": uxgrid.n_face}
    uxda = ux.UxDataArray(data=test_data, dims=dims, uxgrid=uxgrid, name='var2')
    integral = uxda.integrate()
    assert integral.ndim == len(dims) - 1
    nt.assert_almost_equal(integral, np.ones((5, 5)) * 4 * np.pi)


def test_integrate_crashes_when_nnode_equals_nface():
    """Ensure UxDataArray.integrate() crashes for non-face_centered data, even if n_node==n_face.
    regression test for issue #1616.
    """
    # Below is a visualization of the example here, with (ni)=node i; fj=face j:
    # (n1)----------------(n2)
    #  | %%    f1        =%/|
    #  |  %%           =% / |
    #  |   %%       =%   /  |
    #  |    %%   =%  f2 /   |
    #  |     (n3)----(n4) f3|
    #  | f0  %%  =% f4  \   |
    #  |   %%       =%    \ |
    #  | %%   f5       =%  \|
    # (n0)----------------(n5)
    XA, XB, XC, XF = 0, 10, 20, 30
    YA, YB, YF = 0, 10, 25
    node_lonlat = [
        [XA, YA],  # n0
        [XA, YF],  # n1
        [XF, YF],  # n2
        [XB, YB],  # n3
        [XC, YB],  # n4
        [XF, YA],  # n5
    ]
    node_lon = [n[0] for n in node_lonlat]
    node_lat = [n[1] for n in node_lonlat]
    face_node_connectivity = [
        [0, 1, 3],  # f0
        [1, 3, 2],  # f1
        [3, 4, 2],  # f2
        [4, 5, 2],  # f3
        [3, 5, 4],  # f4
        [0, 5, 3],  # f5
    ]
    # convert to xarray
    ds_vars = {
        'node_lon': xr.DataArray(node_lon, dims=['n_node']),
        'node_lat': xr.DataArray(node_lat, dims=['n_node']),
        'face_node_connectivity': xr.DataArray(face_node_connectivity, dims=['n_face', 'n_max_face_nodes']),
    }
    ds = xr.Dataset(ds_vars)
    # convert to uxarray
    uxgrid = ux.Grid(ds, 'UGRID') # putting 'UGRID' avoids raising warning but does not affect results.
    assert uxgrid.n_node == uxgrid.n_face
    # make array of values, convert to uxarray
    vals = xr.DataArray([100,200,300,400,500,600], dims=['n_node'])
    uxarr = ux.UxDataArray(vals, uxgrid=uxgrid)
    # ensure integrate() crashes for non-face_centered data, even if n_node==n_face
    with pytest.raises(ValueError):
        uxarr.integrate()


def _random_uxda(gridpath, shape, dims):
    """Non-uniform test data, so a reordered reduction actually shows up."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    rng = np.random.default_rng(0)
    return ux.UxDataArray(
        data=rng.random(shape), dims=dims, uxgrid=uxgrid, name='var2'
    )


@pytest.mark.parametrize(
    "shape_dims",
    [((5400,), ("n_face",)), ((4, 5, 5400), ("a", "b", "n_face"))],
    ids=["single_dim", "multi_dim"],
)
def test_integrate_returns_uxdataarray_matching_numpy(gridpath, shape_dims):
    """Both branches of integrate() return a UxDataArray matching a plain numpy dot product."""
    shape, dims = shape_dims
    uxda = _random_uxda(gridpath, shape, dims)

    # independent reference, computed without going through integrate()
    expected = uxda.values @ uxda.uxgrid.face_areas.values

    numpy_integral = uxda.integrate()
    dask_integral = uxda.chunk({"n_face": 100}).integrate()

    # backing type is preserved: eager stays eager, chunked stays lazy
    assert isinstance(numpy_integral.data, np.ndarray)
    assert isinstance(dask_integral.data, da.Array)

    for integral in (numpy_integral, dask_integral):
        assert isinstance(integral, ux.UxDataArray)
        assert integral.uxgrid is uxda.uxgrid
        assert integral.dims == dims[:-1]
        nt.assert_allclose(integral.values, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    "shape_dims",
    [((5400,), ("n_face",)), ((4, 5, 5400), ("a", "b", "n_face"))],
    ids=["single_dim", "multi_dim"],
)
def test_integrate_dask_reproduces_numpy_whole_face_dim(gridpath, shape_dims):
    # 'n_face' in a single chunk: xr.dot performs one reduction per block, in the
    # same order as the numpy path's einsum, so agreement must be exact.
    shape, dims = shape_dims
    uxda = _random_uxda(gridpath, shape, dims)

    numpy_integral = uxda.integrate()

    for chunks in ({"n_face": -1}, {"a": 2, "n_face": -1}):
        chunks = {d: s for d, s in chunks.items() if d in dims}
        dask_integral = uxda.chunk(chunks).integrate()

        assert isinstance(dask_integral.data, da.Array)
        assert numpy_integral.dims == dask_integral.dims
        assert numpy_integral.dtype == dask_integral.dtype
        nt.assert_array_equal(numpy_integral.values, dask_integral.values)


@pytest.mark.parametrize(
    "shape_dims",
    [((5400,), ("n_face",)), ((4, 5, 5400), ("a", "b", "n_face"))],
    ids=["single_dim", "multi_dim"],
)
def test_integrate_dask_reproduces_numpy_chunked_face_dim(gridpath, shape_dims):
    # Splitting 'n_face' splits the reduction: xr.dot sums per-chunk partials, so
    # the result differs from the numpy path's single einsum in the last few ULP.
    # This is the one changed routine that is close-but-not-bitwise, hence a
    # tolerance rather than assert_array_equal. Observed worst case is ~2e-15
    # relative on this mesh, independent of chunk size.
    shape, dims = shape_dims
    uxda = _random_uxda(gridpath, shape, dims)

    numpy_integral = uxda.integrate()

    for chunks in ({"n_face": 100}, {"n_face": 1350}, {"n_face": 7}):
        dask_integral = uxda.chunk(chunks).integrate()

        assert isinstance(dask_integral.data, da.Array)
        assert numpy_integral.dims == dask_integral.dims
        assert numpy_integral.dtype == dask_integral.dtype
        nt.assert_allclose(
            numpy_integral.values, dask_integral.values, rtol=1e-12, atol=0.0
        )
