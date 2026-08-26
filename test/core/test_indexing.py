"""
Purpose: tests related to indexing Grid, UxDataArray, and/or UxDataset,
e.g. UxDataArray's and UxDataset's .isel() and .sel() methods.

(Some .isel() and sel() tests are in test_dataarray.py and test_dataset.py,
but could maybe be moved here? Having test_indexing.py as its own file helps
to ensure consistency between UxDataArray and UxDataset indexing.)
"""
import numpy as np
import uxarray as ux

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
