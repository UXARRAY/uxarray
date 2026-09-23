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
    Also contains regression test for #1714.
    """
    # test corresponding to the workflow described in #1641, but for UxDataset
    uxds = ux.tutorial.open_dataset("outCSne30-vortex")
    # (the next few lines, through the `assert`, also serve as a regression test for #1714)
    uxds1 = uxds.assign_coords(n_face=np.arange(uxds.n_face.size))
    uxds2 = uxds1.isel(n_face=range(0, 100, 5))
    uxds3 = uxds2 + 7
    # "check what the results look like on what were originally faces 20, 30, and 40"
    uxds4 = uxds3.sel(n_face=[20,30,40])
    assert uxds4.sizes['n_face'] == uxds4.uxgrid.n_face == 3

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
    for UxDataArrays and UxDatasets. The UxDataset checks serve as a regression test for #1713.
    """
    ds = ux.tutorial.open_dataset("outCSne30-vortex")
    assert "n_face" in ds.dims
    result = ds.isel(n_edge=7)
    assert result.sizes["n_face"] == result.uxgrid.n_face == 2
    result = ds.sel(n_edge=7)
    assert result.sizes["n_face"] == result.uxgrid.n_face == 2

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
    Also contains regression test for #1714.
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
    # (the remaining lines also serve as a regression test for #1714)
    labeled_ds = uxds.assign_coords(n_face=np.arange(grid.n_face))
    result = labeled_ds.sel(n_face=slice(0, 2))
    assert result.n_face.size == result.uxgrid.n_face == 3

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


def test_isel_can_use_bool_with_coords():
    """ensure isel() supports indexing by a boolean indexer array with coords.
    Regression test for bug (1) discovered during review of PR #1759.
    """
    ds = ux.tutorial.open_dataset("quad-hexagon")
    arr = ds['t2m']

    # simplest case: boolean xr.DataArray mask with coords.
    tokeep0 = xr.DataArray([True, False, True, False], coords={'tokeep': [2,4,6,8]})
    # (also want to test UxDataArray boolean mask (ensure no infinite recursion))
    tokeep1 = arr * xr.DataArray([True, False, True, True], dims={'n_face'}) > 0
    assert isinstance(tokeep1, ux.UxDataArray)
    # (also want to test UxDataArray boolean mask with coords
    tokeep2 = tokeep1.assign_coords(tokeep=('n_face', [1,3,5,7]))
    assert isinstance(tokeep2, ux.UxDataArray)
    assert np.all(tokeep2.coords['tokeep'] == [1,3,5,7])

    # test on UxDataArray
    result0 = arr.isel(n_face=tokeep0)
    assert result0.sizes['n_face'] == result0.uxgrid.n_face == 2
    assert np.all(result0.coords['tokeep'] == [2,6])
    result1 = arr.isel(n_face=tokeep1)
    assert result1.sizes['n_face'] == result1.uxgrid.n_face == 3
    result2 = arr.isel(n_face=tokeep2)
    assert result2.sizes['n_face'] == result2.uxgrid.n_face == 3
    assert np.all(result2.coords['tokeep'] == [1,5,7])
    # (reviewer found bug on arr.where(), so doing a spot check for that here too)
    result_where2 = arr.where(tokeep2, drop=True)
    assert result2.equals(result_where2)

    # repeat tests for UxDataset
    result0 = ds.isel(n_face=tokeep0)
    assert result0.sizes['n_face'] == result0.uxgrid.n_face == 2
    assert np.all(result0.coords['tokeep'] == [2,6])
    result1 = ds.isel(n_face=tokeep1)
    assert result1.sizes['n_face'] == result1.uxgrid.n_face == 3
    result2 = ds.isel(n_face=tokeep2)
    assert result2.sizes['n_face'] == result2.uxgrid.n_face == 3
    assert np.all(result2.coords['tokeep'] == [1,5,7])
    result_where2 = ds.where(tokeep2, drop=True)
    assert result2.equals(result_where2)

    # also test second reviewer's example (2) from PR #1759:
    uxds = ux.tutorial.open_dataset("quad-hexagon")
    uxds = uxds.assign_coords(node_id=("n_node", np.arange(uxds.uxgrid.n_node)))
    mask = xr.DataArray([True, False, True, False], dims="n_face",
                        coords={"lab": ("n_face", [10, 20, 30, 40])})
    uxds.isel(n_face=mask)    # (just ensuring it doesn't crash)


def test_indexing_by_dataarray():
    """ensure isel() and sel() with indexer=xr.DataArray(...) both work as expected.
    The dims/coords of the Grid object should never incorporate indexer's dims/coords.

    The dims of the data object (UxDataArray or UxDataset) should not incorporate
    the indexer's dims when indexing along a grid dim (e.g. 'n_face') (this is already true),
    (e.g. don't rename 'n_face' to match the indexer's dim name!) see issue #1712 for details.

    Though, the coords of the data object *should* incorporate the indexer's coords when possible:
        - Always incorporate the indexer's scalar coords.
        - For 1D coords, it is more complicated:
            only incorporate 1D coords if grid dim is 'n_face'
                (because 'n_edge' and 'n_node' indexing don't necessarily lead to same
                size indexer as result)
            and when data is located along 'n_face'
                (because otherwise the inder's dim doesn't align with the result's grid dim).
            I.e., only incorporate 1D coords when indexing face-centered data along 'n_face'.
    See #1712 for more details.

    Regression test for bug in branch (fixed before merging to main) for PR 1729.

    TODO: update accordingly after fixing #1758.
    """
    # --- n_face indexing of n_face data --- #
    # ensure grid's dims/coords do not incorporate indexer's dims/coords:
    indexer0 = xr.DataArray(0, coords={"newcoord": 7})
    indexer1 = xr.DataArray([1,2], dims="newdim", coords={"newdim": [7,8]})
    indexer2 = indexer1.assign_coords({"other1dcoord": ("newdim", [9,10]), "scalarcoord": 100})
    ds = ux.tutorial.open_dataset("quad-hexagon")
    assert "n_face" in ds.dims  # behavior for #1712 depends on data location.
    result0_isel = ds.isel(n_face=indexer0)
    assert "newcoord" not in result0_isel.uxgrid._ds.coords
    assert "newcoord" in result0_isel.coords   # regression test for #1712
    result0_sel = ds.sel(n_face=indexer0)
    assert "newcoord" not in result0_sel.uxgrid._ds.coords
    assert "newcoord" in result0_sel.coords   # regression test for #1712
    result1_isel = ds.isel(n_face=indexer1)
    assert "newdim" not in result1_isel.uxgrid._ds.dims
    assert "newdim" not in result1_isel.uxgrid._ds.coords
    assert "newdim" not in result1_isel.dims and "n_face" in result1_isel.dims  # didn't rename 'n_face'.
    assert "newdim" in result1_isel.coords   # regression test for #1712
    result1_sel = ds.sel(n_face=indexer1)
    assert "newdim" not in result1_sel.uxgrid._ds.dims
    assert "newdim" not in result1_sel.uxgrid._ds.coords
    assert "newdim" not in result1_sel.dims and "n_face" in result1_sel.dims
    assert "newdim" in result1_sel.coords   # regression test for #1712
    result2_isel = ds.isel(n_face=indexer2)
    assert np.all(result2_isel.coords["other1dcoord"] == [9,10])
    assert result2_isel.coords["scalarcoord"] == 100
    result2_sel = ds.sel(n_face=indexer2)
    assert np.all(result2_sel.coords["other1dcoord"] == [9,10])
    assert result2_sel.coords["scalarcoord"] == 100

    # repeat tests but with UxDataArray:
    arr = ds['t2m']
    result0_isel = arr.isel(n_face=indexer0)
    assert "newcoord" not in result0_isel.uxgrid._ds.coords
    assert "newcoord" in result0_isel.coords
    result0_sel = arr.sel(n_face=indexer0)
    assert "newcoord" not in result0_sel.uxgrid._ds.coords
    assert "newcoord" in result0_sel.coords
    result1_isel = arr.isel(n_face=indexer1)
    assert "newdim" not in result1_isel.uxgrid._ds.dims
    assert "newdim" not in result1_isel.uxgrid._ds.coords
    assert "newdim" not in result1_isel.dims and "n_face" in result1_isel.dims
    assert "newdim" in result1_isel.coords
    result1_sel = arr.sel(n_face=indexer1)
    assert "newdim" not in result1_sel.uxgrid._ds.dims
    assert "newdim" not in result1_sel.uxgrid._ds.coords
    assert "newdim" not in result1_sel.dims and "n_face" in result1_sel.dims
    assert "newdim" in result1_sel.coords
    result2_isel = arr.isel(n_face=indexer2)
    assert np.all(result2_isel.coords["other1dcoord"] == [9,10])
    assert result2_isel.coords["scalarcoord"] == 100
    result2_sel = arr.sel(n_face=indexer2)
    assert np.all(result2_sel.coords["other1dcoord"] == [9,10])
    assert result2_sel.coords["scalarcoord"] == 100

    # --- non-n_face indexing and/or non-n_face data --- #
    # using loops to avoid writing extremely long test;
    # loops are slightly harder to debug but worthwhile to include in at least one test,
    # to cover more combinations of cases (e.g., discovered #1758 while making this test).
    ds_face = ds
    ds_node = ux.tutorial.open_dataset("quad-hexagon-random-node")
    #ds_edge = ux.tutorial.open_dataset("quad-hexagon-random-edge")  # uncomment after fixing #1758
    assert "n_face" in ds_face.dims
    assert "n_node" in ds_node.dims
    #assert "n_edge" in ds_edge.dims   # uncomment after fixing #1758
    counter = 0  # (count up during loop to make sure nothing is skipped unexpectedly)
    for method in "isel", "sel":
        for dataset in [ds_face, ds_node]:  # include after fixing #1758
            for grid_dim in ("n_face", "n_edge", "n_node"):
                for to_array in [False, True]:
                    if grid_dim == "n_face" and "n_face" in dataset.dims:
                        continue  # already tested this above!
                    counter += 1
                    obj = dataset[list(dataset.data_vars)[0]] if to_array else dataset
                    result0 = getattr(obj, method)({grid_dim: indexer0})
                    assert "newcoord" not in result0.uxgrid._ds.coords
                    assert "newcoord" in result0.coords  # 0D coord should always show up!
                    result1 = getattr(obj, method)({grid_dim: indexer1})
                    assert "newdim" not in result1.uxgrid._ds.dims
                    assert "newdim" not in result1.uxgrid._ds.coords
                    assert "newdim" not in result1.dims
                    assert "newdim" not in result1.coords
                    result2 = getattr(obj, method)({grid_dim: indexer2})
                    assert "other1dcoord" not in result2.uxgrid._ds.coords
                    assert "other1dcoord" not in result2.coords
                    assert result2.coords["scalarcoord"] == 100
    n_data_grid_dim_combos = 2 * 3 - 1  # after fixing #1758, update to: 3 * 3 - 1.
    # the -1 accounts for skipping when both are "n_face" above.
    assert counter == 2 * n_data_grid_dim_combos * 2


def test_dataset_isel_keeps_bonus_coords():
    """ensure UxDataset.isel() keeps "bonus" coords,
    i.e. coords in the dataset which do not actually appear in any data var.
    Regression test for bug (3) discovered during review of PR #1759.
    """
    ds = ux.tutorial.open_dataset('quad-hexagon')
    ds = ds.assign_coords({'bonus_coord': xr.DataArray(['a', 'b'], dims=['bonus_dim'])})
    assert 'bonus_coord' in ds.coords
    assert all('bonus_coord' not in arr for arr in ds.data_vars.values())
    result = ds.isel(n_face=0)
    assert 'bonus_coord' in result.coords and 'bonus_dim' in result.dims

    # also test using second reviewer's example (1) from PR #1759:
    uxds = ux.tutorial.open_dataset("quad-hexagon")
    uxds = uxds.assign_coords(node_id=("n_node", np.arange(uxds.uxgrid.n_node)))
    sub = uxds.isel(n_face=[0, 1])
    assert 'node_id' in sub.coords


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
    Also includes a regression test for #1714.
    """
    kw_options = ({"method": "nearest"}, {"method": "nearest", "tolerance": 0.1})

    # ---- 1D example ---- #
    # -- UxDataset tests -- #
    ds0 = ux.tutorial.open_dataset("quad-hexagon")
    assert set(ds0.coords) == set()
    assert set(ds0.dims) == {'n_face'}
    ds0_labeled = ds0.assign_coords({'n_face': [0,10,20,30]})

    # (the next few lines also serve as a regression test for #1714)
    ds0.sel(n_face=[0,1])  # (sanity check: no crash when no options provided)
    for kw in kw_options:
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            ds0.sel(n_face=[0,1], **kw)  # provides method, tolerance, or both.
        # separately: checking to ensure that passing these options is fine in "labeled" case.
        ds0_labeled.sel(n_face=[0,10], **kw)

    # ensure same behavior for xarray objects:
    ds0.to_xarray().sel(n_face=[0,1])
    for kw in kw_options:
        with pytest.raises(ValueError, match=r"cannot supply selection options.+for dimension 'n_face'"):
            ds0.to_xarray().sel(n_face=[0,1], **kw)
        ds0_labeled.to_xarray().sel(n_face=[0,10], **kw)

    # ensure supplying just tolerance raises a different error, if indexing is otherwise valid:
    # (the following pytest.raises statement also serves as a regression test for #1714)
    with pytest.raises(ValueError, match=r"tolerance argument only valid if doing.+"):
        ds0_labeled.sel(n_face=[0,10], tolerance=0.1)
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

def test_isel_crash_if_coordinates_conflict():
    """Ensure isel crashes if there is a coordinates conflict,
    such as indexing an array with time dim by an array with a scalar time coord.
    Regression test for bug (2) discovered during review of PR #1759.
    (Also checks indexing an array with scalar time coord by an array with a time dim.)
    """
    ds = ux.tutorial.open_dataset("outCSne30-timeseries")
    arr = ds['psi']
    indexer0 = xr.DataArray(0, coords={'time': arr['time'][0].item()})
    indexer1 = arr.isel(time=0).argmax('n_face')
    indexer2 = xr.DataArray([0,1], dims='time', coords={'time': arr['time'][:2].values})
    assert isinstance(indexer1, ux.UxDataArray)
    assert 'time' in indexer1.coords and 'time' not in indexer1.dims
    MATCH_ERRMSG = "dimension coordinate 'time' conflicts between indexed and indexing objects"
    MATCH_ERRMSG_2 = "The indexer's dimension \('time'\) already exists as a scalar"
    with pytest.raises(IndexError, match=MATCH_ERRMSG):
        arr.to_xarray().isel(n_face=indexer0)  # sanity check that xarray also crashes here.
    with pytest.raises(IndexError, match=MATCH_ERRMSG):
        arr.isel(n_face=indexer0)
    with pytest.raises(IndexError, match=MATCH_ERRMSG):
        arr.to_xarray().isel(n_face=indexer1)  # sanity check that xarray also crashes here.
    with pytest.raises(IndexError, match=MATCH_ERRMSG):
        arr.isel(n_face=indexer1)
    arr_t0 = arr.isel(time=0)
    assert 'time' in arr_t0.coords and 'time' not in arr_t0.dims
    with pytest.raises(ux.errors.DimensionError, match=MATCH_ERRMSG_2):
        arr_t0.isel(n_face=indexer2)

    # repeat tests for UxDataArray:
    with pytest.raises(IndexError, match=MATCH_ERRMSG):
        ds.to_xarray().isel(n_face=indexer0)
    with pytest.raises(IndexError, match=MATCH_ERRMSG):
        ds.isel(n_face=indexer0)
    with pytest.raises(IndexError, match=MATCH_ERRMSG):
        ds.to_xarray().isel(n_face=indexer1)
    with pytest.raises(IndexError, match=MATCH_ERRMSG):
        ds.isel(n_face=indexer1)
    ds_t0 = ds.isel(time=0)
    assert 'time' in ds_t0.coords and 'time' not in ds_t0.dims
    with pytest.raises(ux.errors.DimensionError, match=MATCH_ERRMSG_2):
        ds_t0.isel(n_face=indexer2)

def test_isel_when_indexer_dim_in_uxarray_obj():
    """Ensure isel() crashes with NotImplementedError when the indexer's dim has
    the same name as a dim or non-scalar coordinate in the uxarray object being indexed.
    Regression test for follow-up to bug (2) discovered during review of PR #1759.
    """
    ds = ux.tutorial.open_dataset('quad-hexagon').expand_dims(time=[100,200])
    arr = ds['t2m']
    imax = arr.argmax('n_face')
    assert set(imax.dims) == {'time'} and set(imax.coords) == {'time'}
    assert set(ds.dims) == {'time', 'n_face'} and set(ds.coords) == {'time'}
    assert imax.coords['time'].equals(ds.coords['time'])
    # pure xarray indexing here returns a result with only 'time' dimension;
    # that's the main reason uxarray should raise NotImplementedError in this case.
    xr_result = ds.to_xarray().isel(n_face=imax.to_xarray())
    assert set(xr_result.dims) == {'time'}
    ERRMSG = (
        r"Indexing a {typestr} .+ using an xarray DataArray whose dimension \('time'\) "
        r"is already present .+ is not yet supported"
    )
    with pytest.raises(NotImplementedError, match=ERRMSG.format(typestr="UxDataset")):
        ds.isel(n_face=imax)
    ds1 = ds.swap_dims({'time': 'otherdim'})
    assert 'time' in ds1.coords and 'time' not in ds1.dims
    with pytest.raises(NotImplementedError, match=ERRMSG.format(typestr="UxDataset")):
        ds1.isel(n_face=imax)

    # repeat tests for UxDataArray:
    assert set(arr.dims) == {'time', 'n_face'} and set(arr.coords) == {'time'}
    assert imax.coords['time'].equals(arr.coords['time'])
    xr_result = arr.to_xarray().isel(n_face=imax.to_xarray())
    assert set(xr_result.dims) == {'time'}
    with pytest.raises(NotImplementedError, match=ERRMSG.format(typestr="UxDataArray")):
        arr.isel(n_face=imax)
    arr1 = arr.swap_dims({'time': 'otherdim'})
    assert 'time' in arr1.coords and 'time' not in arr1.dims
    with pytest.raises(NotImplementedError, match=ERRMSG.format(typestr="UxDataArray")):
        arr1.isel(n_face=imax)
