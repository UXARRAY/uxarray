import numpy.testing as nt
import uxarray as ux
import numpy as np
import pytest
import tempfile
import xarray as xr
from pathlib import Path
from unittest.mock import patch
from uxarray.core.utils import _open_dataset_with_fallback
import os

TEST_MESHFILES = Path(__file__).resolve().parent.parent / "meshfiles"

def test_open_geoflow_dataset(gridpath, datasetpath):
    """Loads a single dataset with its grid topology file using uxarray's
    open_dataset call."""

    # Paths to Data Variable files
    data_paths = [
        datasetpath("ugrid", "geoflow-small", "v1.nc"),
        datasetpath("ugrid", "geoflow-small", "v2.nc"),
        datasetpath("ugrid", "geoflow-small", "v3.nc")
    ]

    uxds_v1 = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), data_paths[0])

    # Ideally uxds_v1.uxgrid should NOT be None
    nt.assert_equal(uxds_v1.uxgrid is not None, True)

def test_open_dataset(gridpath, datasetpath, mesh_constants):
    """Loads a single dataset with its grid topology file using uxarray's
    open_dataset call."""

    grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
    data_path = datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc")
    uxds_var2_ne30 = ux.open_dataset(grid_path, data_path)

    nt.assert_equal(uxds_var2_ne30.uxgrid.node_lon.size, mesh_constants['NNODES_outCSne30'])
    nt.assert_equal(len(uxds_var2_ne30.uxgrid._ds.data_vars), mesh_constants['DATAVARS_outCSne30'])
    nt.assert_equal(uxds_var2_ne30.source_datasets, str(data_path))

def test_open_mf_dataset(gridpath, test_data_dir, mesh_constants):
    """Loads multiple datasets with their grid topology file using
    uxarray's open_dataset call."""

    grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
    dsfiles_mf_ne30 = str(test_data_dir) + "/ugrid/outCSne30/outCSne30_*.nc"
    uxds_mf_ne30 = ux.open_mfdataset(grid_path, dsfiles_mf_ne30)

    nt.assert_equal(uxds_mf_ne30.uxgrid.node_lon.size, mesh_constants['NNODES_outCSne30'])
    nt.assert_equal(len(uxds_mf_ne30.uxgrid._ds.data_vars), mesh_constants['DATAVARS_outCSne30'])
    nt.assert_equal(uxds_mf_ne30.source_datasets, dsfiles_mf_ne30)

def test_open_grid(gridpath, mesh_constants):
    """Loads only a grid topology file using uxarray's open_grid call."""
    uxgrid = ux.open_grid(gridpath("ugrid", "geoflow-small", "grid.nc"))

    nt.assert_almost_equal(uxgrid.calculate_total_face_area(), mesh_constants['MESH30_AREA'], decimal=3)

def test_copy_dataset(gridpath, datasetpath):
    """Loads a single dataset with its grid topology file using uxarray's
    open_dataset call and make a copy of the object."""

    uxds_var2_ne30 = ux.open_dataset(
        gridpath("ugrid", "outCSne30", "outCSne30.ug"),
        datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc")
    )

    # make a shallow and deep copy of the dataset object
    uxds_var2_ne30_copy_deep = uxds_var2_ne30.copy(deep=True)
    uxds_var2_ne30_copy = uxds_var2_ne30.copy(deep=False)

    # Ideally uxds_var2_ne30_copy.uxgrid should NOT be None
    nt.assert_equal(uxds_var2_ne30_copy.uxgrid is not None, True)

    # Check that the copy is a shallow copy
    assert uxds_var2_ne30_copy.uxgrid is uxds_var2_ne30.uxgrid
    assert uxds_var2_ne30_copy.uxgrid == uxds_var2_ne30.uxgrid

    # Check that the deep copy is a deep copy
    assert uxds_var2_ne30_copy_deep.uxgrid == uxds_var2_ne30.uxgrid
    assert uxds_var2_ne30_copy_deep.uxgrid is not uxds_var2_ne30.uxgrid

def test_copy_dataarray(gridpath, datasetpath):
    """Loads an unstructured grid and data using uxarray's open_dataset
    call and make a copy of the dataarray object."""

    # Paths to Data Variable files
    data_paths = [
        datasetpath("ugrid", "geoflow-small", "v1.nc"),
        datasetpath("ugrid", "geoflow-small", "v2.nc"),
        datasetpath("ugrid", "geoflow-small", "v3.nc")
    ]

    uxds_v1 = ux.open_dataset(gridpath("ugrid", "geoflow-small", "grid.nc"), data_paths[0])

    # get the uxdataarray object
    v1_uxdata_array = uxds_v1['v1']

    # make a shallow and deep copy of the dataarray object
    v1_uxdata_array_copy_deep = v1_uxdata_array.copy(deep=True)
    v1_uxdata_array_copy = v1_uxdata_array.copy(deep=False)

    # Check that the copy is a shallow copy
    assert v1_uxdata_array_copy.uxgrid is v1_uxdata_array.uxgrid
    assert v1_uxdata_array_copy.uxgrid == v1_uxdata_array.uxgrid

    # Check that the deep copy is a deep copy
    assert v1_uxdata_array_copy_deep.uxgrid == v1_uxdata_array.uxgrid
    assert v1_uxdata_array_copy_deep.uxgrid is not v1_uxdata_array.uxgrid

def test_open_dataset_grid_kwargs(gridpath, datasetpath):
    """Drops ``Mesh2_face_nodes`` from the inputted grid file using
    ``grid_kwargs``"""

    with pytest.raises(ValueError):
        # attempt to open a dataset after dropping face nodes should raise a KeyError
        uxds = ux.open_dataset(
            gridpath("ugrid", "outCSne30", "outCSne30.ug"),
            datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc"),
            grid_kwargs={"drop_variables": "Mesh2_face_nodes"}
                )


def test_open_dataset_with_fallback():
    """Test that the fallback mechanism works when the default engine fails."""

    tmp_path = ""
    ds = None
    ds_fallback = None
    try:
        # Create a simple test dataset
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            data = xr.Dataset({'temp': (['x', 'y'], np.random.rand(5, 5))})
            data.to_netcdf(tmp.name)
            tmp_path = tmp.name

        # Test normal case
        ds = _open_dataset_with_fallback(tmp_path)
        assert isinstance(ds, xr.Dataset)
        assert 'temp' in ds.data_vars

        # Test fallback mechanism with mocked failure
        original_open = xr.open_dataset
        call_count = 0
        def mock_open_dataset(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and 'engine' not in kwargs:
                raise Exception("Simulated engine failure")
            return original_open(*args, **kwargs)

        with patch('uxarray.core.utils.xr.open_dataset', side_effect=mock_open_dataset):
            ds_fallback = _open_dataset_with_fallback(tmp_path)
            assert isinstance(ds_fallback, xr.Dataset)
            assert call_count == 2  # First failed, second succeeded

    finally:
        if ds is not None:
            ds.close()
        if ds_fallback is not None:
            ds_fallback.close()
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_list_grid_names_multigrid(gridpath):
    """List grids from an OASIS-style multi-grid file."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    grid_names = ux.list_grid_names(grid_file)

    assert isinstance(grid_names, list)
    assert set(grid_names) == {"ocn", "atm"}


def test_list_grid_names_single_scrip():
    """List grids from a standard single-grid SCRIP file."""
    grid_path = TEST_MESHFILES / "scrip" / "outCSne8" / "outCSne8.nc"
    grid_names = ux.list_grid_names(grid_path)

    assert isinstance(grid_names, list)
    assert grid_names == ["grid"]


def test_open_multigrid_all_grids(gridpath):
    """Open all grids from a multi-grid file."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    grids = ux.open_multigrid(grid_file)

    assert isinstance(grids, dict)
    assert set(grids.keys()) == {"ocn", "atm"}
    assert grids["ocn"].n_face == 12
    assert grids["atm"].n_face == 20


def test_open_multigrid_specific_grids(gridpath):
    """Open a subset of grids from a multi-grid file."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    grids = ux.open_multigrid(grid_file, gridnames=["ocn"])

    assert set(grids.keys()) == {"ocn"}
    assert grids["ocn"].n_face == 12


def test_open_multigrid_with_masks(gridpath):
    """Open grids with a companion mask file."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    mask_file = gridpath("scrip", "oasis", "masks.nc")

    grids = ux.open_multigrid(grid_file, mask_filename=mask_file)

    assert grids["ocn"].n_face == 8
    assert grids["atm"].n_face == 20


def test_open_multigrid_missing_grid_error(gridpath):
    """Requesting a missing grid should raise."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")

    with pytest.raises(ValueError, match="Grid 'land' not found"):
        ux.open_multigrid(grid_file, gridnames=["land"])
