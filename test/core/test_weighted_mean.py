
import pytest

import dask.array as da
import numpy as np
import numpy.testing as nt

import uxarray as ux
import xarray as xr

def test_quad_hex_face_centered(gridpath, datasetpath):
    """Compares the weighted average computation for the quad hexagon grid
    using a face centered data variable to the expected value computed by
    hand."""
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))

    # expected weighted average computed by hand
    expected_weighted_mean = 297.55

    # compute the weighted mean
    result = uxds['t2m'].weighted_mean()

    # ensure values are within 3 decimal points of each other
    nt.assert_almost_equal(result.values, expected_weighted_mean, decimal=3)

    # check can compute weighted mean from UxDataset too:
    result_ds = uxds.weighted_mean()
    assert result_ds['t2m'].equals(result)

def test_quad_hex_face_centered_dask(gridpath, datasetpath):
    """Compares the weighted average computation for the quad hexagon grid
    using a face centered data variable on a dask-backed UxDataset & Grid to the expected value computed by
    hand."""
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), datasetpath("ugrid", "quad-hexagon", "data.nc"))

    # data to be dask
    uxda = uxds['t2m'].chunk(n_face=1)

    # weights to be dask
    uxda.uxgrid.face_areas = uxda.uxgrid.face_areas.chunk(n_face=1)

    # create lazy result
    lazy_result = uxda.weighted_mean()

    assert isinstance(lazy_result.data, da.Array)

    # compute result
    computed_result = lazy_result.compute()

    assert isinstance(computed_result.data, np.ndarray)

    expected_weighted_mean = 297.55

    # ensure values are within 3 decimal points of each other
    nt.assert_almost_equal(computed_result.values, expected_weighted_mean, decimal=3)

    # check can compute weighted mean from UxDataset too:
    result_ds = uxds.weighted_mean().compute()
    assert result_ds['t2m'].equals(computed_result)

def test_quad_hex_edge_centered(gridpath, test_data_dir):
    """Compares the weighted average computation for the quad hexagon grid
    using an edge centered data variable to the expected value computed by
    hand."""
    quad_hex_data_path_edge_centered = test_data_dir / "ugrid" / "quad-hexagon" / "random-edge-data.nc"
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), quad_hex_data_path_edge_centered)

    # expected weighted average computed by hand
    expected_weighted_mean = (uxds['random_data_edge'].values * uxds.uxgrid.edge_node_distances).sum() / uxds.uxgrid.edge_node_distances.sum()

    # compute the weighted mean
    result = uxds['random_data_edge'].weighted_mean()

    nt.assert_equal(result, expected_weighted_mean)

    # check can compute weighted mean from UxDataset too:
    result_ds = uxds.weighted_mean()
    assert result_ds['random_data_edge'].equals(result)

def test_quad_hex_edge_centered_dask(gridpath, test_data_dir):
    """Compares the weighted average computation for the quad hexagon grid
    using an edge centered data variable on a dask-backed UxDataset & Grid to the expected value computed by
    hand."""
    quad_hex_data_path_edge_centered = test_data_dir / "ugrid" / "quad-hexagon" / "random-edge-data.nc"
    uxds = ux.open_dataset(gridpath("ugrid", "quad-hexagon", "grid.nc"), quad_hex_data_path_edge_centered)

    # data to be dask
    uxda = uxds['random_data_edge'].chunk(n_edge=1)

    # weights to be dask
    uxda.uxgrid.edge_node_distances = uxda.uxgrid.edge_node_distances.chunk(n_edge=1)

    # create lazy result
    lazy_result = uxds['random_data_edge'].weighted_mean()

    assert isinstance(lazy_result.data, da.Array)

    # compute result
    computed_result = lazy_result.compute()

    assert isinstance(computed_result.data, np.ndarray)

    # expected weighted average computed by hand
    expected_weighted_mean = (uxds['random_data_edge'].values * uxds.uxgrid.edge_node_distances).sum() / uxds.uxgrid.edge_node_distances.sum()

    # ensure values are within 3 decimal points of each other
    nt.assert_almost_equal(computed_result.values, expected_weighted_mean, decimal=3)

    # check can compute weighted mean from UxDataset too:
    result_ds = uxds.weighted_mean().compute()
    assert result_ds['random_data_edge'].equals(computed_result)

def test_weighted_mean_crash_if_node_centered_and_no_weights():
    """Ensure weighted_mean crashes if the data is node-centered and no weights are provided.
    Regression test for bug 2 of issue #1800.
    """
    ds = ux.tutorial.open_dataset('quad-hexagon-random-node')
    arr = ds['random_data_node']
    ERRMSG_ARR = r"weighted_mean\(\) cannot automatically infer weights for node-centered data."
    with pytest.raises(ux.errors.DataCenteringError, match=ERRMSG_ARR):
        arr.weighted_mean()
    ERRMSG_DS = r"Expected UxDataset\.dims to contain at least 1 grid dimension supported by 'weighted_mean'"
    with pytest.raises(ux.errors.DataCenteringError, match=ERRMSG_DS):
        ds.weighted_mean()

def test_csne30_equal_area(gridpath, datasetpath):
    """Compute the weighted average with a grid that has equal-area faces and
    compare the result to the regular mean."""
    uxds = ux.open_dataset(
        gridpath("ugrid", "outCSne30", "outCSne30.ug"),
        datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
    )
    face_areas = uxds.uxgrid.face_areas

    # set the area of each face to be one
    uxds.uxgrid._ds['face_areas'].data = np.ones(uxds.uxgrid.n_face)

    weighted_mean = uxds['psi'].weighted_mean()
    unweighted_mean = uxds['psi'].mean()

    # with equal area, both should be equal
    nt.assert_almost_equal(weighted_mean, unweighted_mean)

@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_csne30_equal_area_dask(gridpath, datasetpath, chunk_size):
    """Compares the weighted average computation for the quad hexagon grid
        using a face centered data variable on a dask-backed UxDataset & Grid to the expected value computed by
        hand."""
    uxds = ux.open_dataset(
        gridpath("ugrid", "outCSne30", "outCSne30.ug"),
        datasetpath("ugrid", "outCSne30", "outCSne30_vortex.nc")
    )

    # data and weights to be dask
    uxda = uxds['psi'].chunk(n_face=chunk_size)
    uxda.uxgrid.face_areas = uxda.uxgrid.face_areas.chunk(n_face=chunk_size)

    # Calculate lazy result
    lazy_result = uxds['psi'].weighted_mean()
    assert isinstance(lazy_result.data, da.Array)

    # compute result
    computed_result = lazy_result.compute()
    assert isinstance(computed_result.data, np.ndarray)

    # expected weighted average computed by hand
    expected_weighted_mean = (uxds['psi'].values * uxds.uxgrid.face_areas).sum() / uxds.uxgrid.face_areas.sum()

    # ensure values are within 3 decimal points of each other
    nt.assert_almost_equal(computed_result.values, expected_weighted_mean, decimal=3)

def test_weighted_mean_if_provided_weights():
    """Ensure can provide weights to weighted_mean and get correct results."""
    ds_face = ux.tutorial.open_dataset('quad-hexagon-random-face')
    ds_node = ux.tutorial.open_dataset('quad-hexagon-random-node')
    ds_edge = ux.tutorial.open_dataset('quad-hexagon-random-edge')
    for ds in [ds_face, ds_node, ds_edge]:
        assert len(ds.dims) == 1 and len(ds.data_vars) == 1
        arr = ds.to_array('variable').isel(variable=0)
        weights_values = np.arange(arr.size)
        weights_nparr = weights_values
        weights_list = list(weights_nparr)
        weights_da = xr.DataArray(weights_nparr, dims=arr.dims)
        expected_weighted_mean = (arr.values * weights_values).sum() / weights_values.sum()
        # ^just .sum() is fine because arr is 1D; don't need to worry about other dims.
        for weights in [weights_nparr, weights_list, weights_da]:
            weighted_mean = arr.weighted_mean(weights=weights)
            assert weighted_mean.ndim == 0   # arr is 1D; now no dims remain.
            nt.assert_equal(weighted_mean.item(), expected_weighted_mean)
        # repeat checks but for UxDataset:
        for weights in [weights_nparr, weights_list, weights_da]:
            weighted_mean_ds = ds.weighted_mean(weights=weights)
            weighted_mean = weighted_mean_ds.to_array('variable').isel(variable=0)
            assert weighted_mean.ndim == 0
            nt.assert_equal(weighted_mean.item(), expected_weighted_mean)
        ## special cases for xr.DataArray and/or chunked weights:
        # wrong dim --> need to crash.
        weights_da_wrongdim = xr.DataArray(weights_nparr, dims=['wrongdim'])
        with pytest.raises(ux.errors.DimensionError):
            arr.weighted_mean(weights=weights_da_wrongdim)
        with pytest.raises(ux.errors.DimensionError):
            ds.weighted_mean(weights=weights_da_wrongdim)
        # has a scalar coord --> result should have that coord, too.
        weights_da_scalarcoord = xr.DataArray(weights_nparr, dims=arr.dims, coords={'scalarcoord': 7})
        weighted_mean = arr.weighted_mean(weights=weights_da_scalarcoord)
        assert weighted_mean.coords['scalarcoord'] == 7
        weighted_mean_ds = ds.weighted_mean(weights=weights_da_scalarcoord)
        assert weighted_mean_ds.coords['scalarcoord'] == 7
        # has grid_dim coords which disagree with arr's coords --> need to crash.
        grid_dim = arr._grid_dim
        dsC = ds.assign_coords({grid_dim: 10*np.arange(ds.sizes[grid_dim])})
        arrC = dsC.to_array('variable').isel(variable=0)
        weightsC0 = arrC.to_xarray().assign_coords({grid_dim: 100*np.arange(arrC.sizes[grid_dim])})
        weightsC1 = arrC.to_xarray().isel({grid_dim: [0,1]})
        with pytest.raises(ux.errors.DimensionError):
            arrC.weighted_mean(weights=weightsC0)
        with pytest.raises(ux.errors.DimensionError):
            dsC.weighted_mean(weights=weightsC0)
        with pytest.raises(ux.errors.DimensionError):
            arrC.weighted_mean(weights=weightsC1)
        with pytest.raises(ux.errors.DimensionError):
            dsC.weighted_mean(weights=weightsC1)
        # chunked weights and/or chunked input --> chunked result
        dsc = ds.chunk({ds._grid_dim: -1})
        arrc = dsc.to_array('variable').isel(variable=0)
        weightsc_xr = weights_da.chunk({ds._grid_dim: -1})
        weightsc_da = weightsc_xr.data   # weights as dask array
        assert arrc.weighted_mean(weights=weights_da).chunks is not None
        assert arrc.weighted_mean(weights=weightsc_xr).chunks is not None
        assert arrc.weighted_mean(weights=weightsc_da).chunks is not None
        assert arr.weighted_mean(weights=weightsc_xr).chunks is not None
        assert arr.weighted_mean(weights=weightsc_da).chunks is not None
        assert arr.weighted_mean(weights=weights_da).chunks is None
        var = list(ds.data_vars)[0]
        assert dsc.weighted_mean(weights=weights_da)[var].chunks is not None
        assert dsc.weighted_mean(weights=weightsc_xr)[var].chunks is not None
        assert dsc.weighted_mean(weights=weightsc_da)[var].chunks is not None
        assert ds.weighted_mean(weights=weightsc_xr)[var].chunks is not None
        assert ds.weighted_mean(weights=weightsc_da)[var].chunks is not None
        assert ds.weighted_mean(weights=weights_da)[var].chunks is None


def test_weighted_mean_doesnt_care_about_dim_order():
    """Ensure weighted mean does not care about dimension order.
    E.g., grid dim doesn't need to be the last dim, in order to get correct results.
    Regression test for bug 1 of issue #1800.
    """
    arr = ux.tutorial.open_dataset("outCSne30-timeseries")['psi']
    assert arr.dims == ('time', 'n_face')   # (assert original has 'n_face' dim last.)
    order0_result = arr.weighted_mean()
    order1_result = arr.transpose('n_face', 'time').weighted_mean()
    assert order0_result.dims == ('time',) == order1_result.dims
    np.allclose(order0_result, order1_result, atol=0, rtol=1e-13)
    # ensure correctness even if two dims have same size:
    n_time = arr.sizes['time']
    sliced = arr.isel(n_face=slice(n_time))
    assert sliced.sizes == {'time': n_time, 'n_face': n_time}
    order0_result = sliced.weighted_mean()
    order1_result = sliced.transpose('n_face', 'time').weighted_mean()
    assert order0_result.dims == ('time',) == order1_result.dims
    np.allclose(order0_result, order1_result, atol=0, rtol=1e-13)
