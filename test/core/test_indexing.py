"""
Purpose: tests related to indexing Grid, UxDataArray, and/or UxDataset,
e.g. UxDataArray's and UxDataset's .isel() and .sel() methods.

(Some .isel() and sel() tests are in test_dataarray.py and test_dataset.py,
but could maybe be moved here? Having test_indexing.py as its own file helps
to ensure consistency between UxDataArray and UxDataset indexing.)
"""
import numpy as np
import pytest
import uxarray as ux
import xarray as xr

def test_sel_indexes_grid():
    """ensure obj.sel({grid_dim: ...}) actually indexes the result.uxgrid, too,
    for UxDataArrays and UxDatasets. Regression test for #1641.
    """
    # extremely simple case:
    uxds = ux.tutorial.open_dataset("quad-hexagon")
    result = uxds.sel(n_face=0)
    assert result.sizes['n_face'] == result.uxgrid.n_face == 1
    # (similar check for UxDataArray)
    uxarr = uxds['t2m']
    result = uxarr.sel(n_face=0)
    assert result.sizes['n_face'] == result.uxgrid.n_face == 1

    # more complicated case:
    uxds = ux.tutorial.open_dataset("outCSne30-timeseries")
    assert uxds.sizes['n_face'] > 5000
    result = uxds.sel(time=['2018-04-28T00', '2018-04-28T03'], n_face=range(0, 5000, 100))
    assert result.sizes == {'time': 2, 'n_face': 50}
    assert result.uxgrid.n_face == 50
    # (similar check for UxDataArray)
    uxarr = uxds['psi']
    result = uxarr.sel(time=['2018-04-28T00', '2018-04-28T03'], n_face=range(0, 5000, 100))
    assert result.sizes == {'time': 2, 'n_face': 50}
    assert result.uxgrid.n_face == 50

def test_sel_uses_grid_dim_labels():
    """ensure obj.sel({grid_dim: ...}) actually utilizes coordinate labels on that grid dim,
    for UxDataArrays and UxDatasets. Regression test for #1641.
    TODO: fix #1714 then uncomment the UxDataset tests below
    """
    # test corresponding to the workflow described in #1641, but for UxDataset
    uxds = ux.tutorial.open_dataset("outCSne30-vortex")
    # (uncomment the next few lines after fixing #1714)
    # uxds1 = uxds.assign_coords(n_face=np.arange(uxds.n_face.size))
    # uxds2 = uxds1.isel(n_face=range(0, 100, 5))
    # uxds3 = uxds2 + 7
    # # "check what the results look like on what were originally faces 20, 30, and 40"
    # uxds4 = uxds3.sel(n_face=[20,30,40])
    # assert uxds4.sizes['n_face'] == uxds4.uxgrid.n_face == 3

    # test corresponding to the workflow described in #1641, for UxDataArray
    uxarr = uxds['psi']
    uxarr1 = uxarr.assign_coords(n_face=np.arange(uxarr.n_face.size))
    uxarr2 = uxarr1.isel(n_face=range(0, 100, 5))
    uxarr3 = uxarr2 + 7
    # "check what the results look like on what were originally faces 20, 30, and 40"
    uxarr4 = uxarr3.sel(n_face=[20,30,40])
    assert uxarr4.sizes['n_face'] == uxarr4.uxgrid.n_face == 3

    # test to check what happens if using coordinate labels not equal to indexes:
    uxarr1 = uxarr.assign_coords(n_face=10*np.arange(uxarr.n_face.size))
    uxarr2 = uxarr1.isel(n_face=[5,6,7,8])
    assert np.all(uxarr2 == uxarr1.sel(n_face=[50,60,70,80]))
    assert np.all(uxarr2['n_face'] == [50,60,70,80])  # (isel shouldn't drop coord labels)
    uxarr3 = uxarr2.isel(n_face=2)
    assert np.all(uxarr3 == uxarr2.sel(n_face=70))

def test_can_index_grid_dim_not_in_data():
    """ensure isel() and sel() can both index a grid dim even if that dim is not present in the data itself;
    for UxDataArrays and UxDatasets. TODO: fix #1713 then uncomment the UxDataset tests below.
    """
    ds = ux.tutorial.open_dataset("outCSne30-vortex")
    # (uncomment the next few lines after fixing #1713)
    # assert "n_face" in ds.dims
    # result = ds.isel(n_edge=7)
    # assert result.sizes["n_face"] == result.uxgrid.n_face == 2
    # result = ds.sel(n_edge=7)
    # assert result.sizes["n_face"] == result.uxgrid.n_face == 2

    arr = ds["psi"]
    result = arr.isel(n_edge=7)
    assert result.sizes["n_face"] == result.uxgrid.n_face == 2
    result = arr.sel(n_edge=7)
    assert result.sizes["n_face"] == result.uxgrid.n_face == 2

def test_isel_can_use_slice():
    """ensure isel() can use slice() objects as indexers, and provides expected results,
    with expected sizes, for UxDataArrays and UxDatasets.
    Regression test for #1639.
    """
    ds = ux.tutorial.open_dataset("outCSne30-vortex")
    result = ds.isel(n_face=slice(None, None, 10))  # should get every 10th face.
    assert result.sizes['n_face'] == result.uxgrid.n_face == ds.sizes['n_face'] // 10
    result = ds.isel(n_face=slice(2, 15, 3))  # should get faces 2, 5, 8, 11, 14
    assert result.sizes['n_face'] == result.uxgrid.n_face == 5
    assert result.equals(ds.isel(n_face=[2,5,8,11,14]))

    # repeat tests but with UxDataArray:
    arr = ds['psi']
    result = arr.isel(n_face=slice(None, None, 10))  # should get every 10th face.
    assert result.sizes['n_face'] == result.uxgrid.n_face == arr.sizes['n_face'] // 10
    result = arr.isel(n_face=slice(2, 15, 3))
    assert result.sizes['n_face'] == result.uxgrid.n_face == 5
    assert result.equals(arr.isel(n_face=[2,5,8,11,14]))

def test_sel_can_use_slice():
    """ensure sel() can use slice() objects as indexers, and provides expected results,
    with expected sizes, for UxDataArrays and UxDatasets.
    Regression test inspired by reviewer comment in #1641, also related to #1639.
    TODO: fix #1714 then uncomment the relevant UxDataset tests below
    """
    grid = ux.Grid.from_healpix(zoom=0)  # 12 faces
    arr = ux.UxDataArray(
        np.arange(grid.n_face, dtype=float), dims="n_face", uxgrid=grid
    )
    # check with unlabeled data:
    result = arr.sel(n_face=slice(0, 2))
    assert result.n_face.size == result.uxgrid.n_face == 2
    # check with labeled data (includes both endpoints,
    #   as per docstring and in agreement with xarray behavior)
    labeled = arr.assign_coords(n_face=np.arange(grid.n_face))
    result = labeled.sel(n_face=slice(0, 2))
    assert result.n_face.size == result.uxgrid.n_face == 3

    # repeat test above but with UxDataset:
    uxds = ux.UxDataset({'data': arr.to_xarray()}, uxgrid=grid)
    result = uxds.sel(n_face=slice(0, 2))
    assert result.n_face.size == result.uxgrid.n_face == 2
    # (uncomment the next few lines after fixing #1714)
    # labeled_ds = uxds.assign_coords(n_face=np.arange(grid.n_face))
    # result = labeled_ds.sel(n_face=slice(0, 2))
    # assert result.n_face.size == result.uxgrid.n_face == 3

def test_isel_can_use_bool():
    """ensure isel() supports indexing by a boolean indexer array.
    Regression test for #1728.
    """
    ds = ux.tutorial.open_dataset("quad-hexagon")
    assert ds.isel(n_face=[True, False, False, False]).equals(ds.isel(n_face=0))
    assert ds.isel(n_face=[False, True, False, True]).equals(ds.isel(n_face=[1,3]))
    result = ds.isel(n_face=[False, False, False, False])
    assert result.sizes['n_face'] == result.uxgrid.n_face == 0

    # repeat tests but with UxDataArray:
    arr = ds['t2m']
    assert arr.isel(n_face=[True, False, False, False]).equals(arr.isel(n_face=0))
    assert arr.isel(n_face=[False, True, False, True]).equals(arr.isel(n_face=[1,3]))
    result = arr.isel(n_face=[False, False, False, False])
    assert result.sizes['n_face'] == result.uxgrid.n_face == 0

def test_indexing_does_not_edit_indexers_dict():
    """ensure isel() and sel() do not edit the provided indexers dict.
    Regression test for #1711.
    """
    ds = ux.tutorial.open_dataset('quad-hexagon')
    choices = {'n_face': 0}
    resultA = ds.isel(choices)
    assert choices == {'n_face': 0}   # calling isel() should not modify the inputs!
    resultB = ds.isel(choices)
    assert resultA.equals(resultB)
    resultA_sel = ds.sel(choices)
    assert choices == {'n_face': 0}   # calling sel() should not modify the inputs!
    resultB_sel = ds.sel(choices)
    assert resultA_sel.equals(resultB_sel)

    # repeat tests but with UxDataArray:
    arr = ds['t2m']
    choices = {'n_face': 0}
    resultA = arr.isel(choices)
    assert choices == {'n_face': 0}
    resultB = arr.isel(choices)
    assert resultA.equals(resultB)
    resultA_sel = arr.sel(choices)
    assert choices == {'n_face': 0}
    resultB_sel = arr.sel(choices)
    assert resultA_sel.equals(resultB_sel)


# ------- tests related to error handling ------- #

def test_isel_crash_if_2d_indexer():
    """ensure isel() crashes if an indexer along a grid dimension is 2D (or more)."""
    ds = ux.tutorial.open_dataset("quad-hexagon")
    clever_indexer = xr.DataArray([[0,1,1],[2,3,3]], dims=["newdimA","newdimB"])
    # (ensure clever_indexer is actually valid for xarray indexing purposes,
    # otherwise the uxarray test would not be particularly meaningful.)
    _tmp = ds.to_xarray().isel(n_face=clever_indexer)
    assert _tmp.sizes == {'newdimA': 2, 'newdimB': 3}
    assert _tmp.isel(newdimA=1, newdimB=0).equals(ds.to_xarray().isel(n_face=2))
    # (now actually make sure that uxarray crashes with the same indexer)
    with pytest.raises(ux.errors.DimensionError):
        ds.isel(n_face=clever_indexer)

    # repeat tests but with UxDataArray (no need to repeat the indexer check though)
    arr = ds['t2m']
    with pytest.raises(ux.errors.DimensionError):
        arr.isel(n_face=clever_indexer)

def test_sel_crash_if_provided_selection_options_with_coordless_dims():
    """ensure sel() crashes if providing `tolerance` and/or `method` options
    whenever any of the indexed dims have no associated coordinates.
    (Tests below also demonstrate that this behavior is consistent with xarray.)
    Regression test inspired by reviewer comment in #1641.
    TODO: fix #1714 then uncomment the relevant UxDataset tests below
    """
    kw_options = ({"method": "nearest"}, {"method": "nearest", "tolerance": 0.1})

    # ---- 1D example ---- #
    # -- UxDataset tests -- #
    ds0 = ux.tutorial.open_dataset("quad-hexagon")
    assert set(ds0.coords) == set()
    assert set(ds0.dims) == {'n_face'}
    ds0_labeled = ds0.assign_coords({'n_face': [0,10,20,30]})

    # (uncomment the next few lines after fixing #1714)
    # ds0.sel(n_face=[0,1])  # (sanity check: no crash when no options provided)
    # for kw in kw_options:
    #     with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
    #         ds0.sel(n_face=[0,1], **kw)  # provides method, tolerance, or both.
    #     # separately: checking to ensure that passing these options is fine in "labeled" case.
    #     ds0_labeled.sel(n_face=[0,10], **kw)

    # ensure same behavior for xarray objects:
    ds0.to_xarray().sel(n_face=[0,1])
    for kw in kw_options:
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            ds0.to_xarray().sel(n_face=[0,1], **kw)
        ds0_labeled.to_xarray().sel(n_face=[0,10], **kw)

    # ensure supplying just tolerance raises a different error, if indexing is otherwise valid:
    # (uncomment the next few lines after fixing #1714)
    # with pytest.raises(ValueError, match=r"tolerance argument only valid if doing.+"):
    #     ds0_labeled.sel(n_face=[0,10], tolerance=0.1)
    with pytest.raises(ValueError, match=r"tolerance argument only valid if doing.+"):
        ds0_labeled.to_xarray().sel(n_face=[0,10], tolerance=0.1)

    # -- UxDataArray tests -- #
    # (like above, but for UxDataArray objects. Fewer comments; see comments above.)
    arr0 = ds0['t2m']
    arr0_labeled = ds0_labeled['t2m']
    arr0.sel(n_face=[0,1])
    for kw in kw_options:
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            arr0.sel(n_face=[0,1], **kw)
        arr0_labeled.sel(n_face=[0,10], **kw)

    arr0.to_xarray().sel(n_face=[0,1])
    for kw in kw_options:
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            arr0.to_xarray().sel(n_face=[0,1], **kw)
        arr0_labeled.to_xarray().sel(n_face=[0,10], **kw)

    with pytest.raises(ValueError, match=r"tolerance argument only valid if doing.+"):
        arr0_labeled.sel(n_face=[0,10], tolerance=0.1)
    with pytest.raises(ValueError, match=r"tolerance argument only valid if doing.+"):
        arr0_labeled.to_xarray().sel(n_face=[0,10], tolerance=0.1)

    # ---- 2D example ---- #
    # -- UxDataset tests -- #
    ds1_labeled = ux.tutorial.open_dataset("outCSne30-timeseries")
    assert set(ds1_labeled.coords) == {'time'}
    assert set(ds1_labeled.dims) == {'time', 'n_face'}
    ds1 = ds1_labeled.drop_vars('time')

    # (sanity checks: no crash when no options provided)
    ds1_labeled.sel(time='2018-04-28T02')
    ds1_labeled.sel(time='2018-04-28T02', n_face=[0,1])
    ds1_labeled.sel(n_face=2)
    ds1.sel(time=4)
    ds1.sel(time=4, n_face=[3])
    # loop with options
    for kw in kw_options:
        # passing options is fine when all indexed dims have coordinates.
        ds1_labeled.sel(time='2018-04-28T02', **kw)
        # (otherwise, should crash!)
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            ds1_labeled.sel(time='2018-04-28T02', n_face=[0,1], **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            ds1.sel(n_face=2, **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'time'"):
            ds1.sel(time=4, **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options"):
            # (message might mention either dimension in this case)
            ds1.sel(time=4, n_face=[3], **kw)

    # ensure same behavior for xarray objects:
    ds1_labeled.to_xarray().sel(time='2018-04-28T02')
    ds1_labeled.to_xarray().sel(time='2018-04-28T02', n_face=[0,1])
    ds1_labeled.to_xarray().sel(n_face=2)
    ds1.to_xarray().sel(time=4)
    ds1.to_xarray().sel(time=4, n_face=[3])
    for kw in kw_options:
        ds1_labeled.to_xarray().sel(time='2018-04-28T02', **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            ds1_labeled.to_xarray().sel(time='2018-04-28T02', n_face=[0,1], **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            ds1.to_xarray().sel(n_face=2, **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'time'"):
            ds1.to_xarray().sel(time=4, **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options"):
            ds1.to_xarray().sel(time=4, n_face=[3], **kw)

    # -- UxDataArray tests -- #
    # (like above, but for UxDataArray objects. Fewer comments; see comments above.)
    arr1_labeled = ds1_labeled['psi']
    arr1 = ds1['psi']

    arr1_labeled.sel(time='2018-04-28T02')
    arr1_labeled.sel(time='2018-04-28T02', n_face=[0,1])
    arr1_labeled.sel(n_face=2)
    arr1.sel(time=4)
    arr1.sel(time=4, n_face=[3])
    for kw in kw_options:
        arr1_labeled.sel(time='2018-04-28T02', **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            arr1_labeled.sel(time='2018-04-28T02', n_face=[0,1], **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            arr1.sel(n_face=2, **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'time'"):
            arr1.sel(time=4, **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options"):
            arr1.sel(time=4, n_face=[3], **kw)

    arr1_labeled.to_xarray().sel(time='2018-04-28T02')
    arr1_labeled.to_xarray().sel(time='2018-04-28T02', n_face=[0,1])
    arr1_labeled.to_xarray().sel(n_face=2)
    arr1.to_xarray().sel(time=4)
    arr1.to_xarray().sel(time=4, n_face=[3])
    for kw in kw_options:
        arr1_labeled.to_xarray().sel(time='2018-04-28T02', **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            arr1_labeled.to_xarray().sel(time='2018-04-28T02', n_face=[0,1], **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            arr1.to_xarray().sel(n_face=2, **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'time'"):
            arr1.to_xarray().sel(time=4, **kw)
        with pytest.raises(ValueError, match=r"cannot supply selection options"):
            arr1.to_xarray().sel(time=4, n_face=[3], **kw)
