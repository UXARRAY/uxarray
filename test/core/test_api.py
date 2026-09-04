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


def test_open_dataset_single_combined_mpas_file(gridpath):
    """Loads a combined MPAS grid-and-data file with a single argument."""

    # Use a known combined grid-and-data MPAS file.
    file_path = gridpath("mpas", "QU", "oQU480.231010.nc")

    uxds_single = ux.open_dataset(file_path)
    uxds_pair = ux.open_dataset(file_path, file_path)

    # Ensure that the single-argument path actually loads data variables
    assert len(uxds_single.data_vars) > 0
    nt.assert_equal(uxds_single.uxgrid.source_grid_spec, "MPAS")
    nt.assert_equal(uxds_single.source_datasets, str(file_path))
    nt.assert_equal(uxds_single.sizes["n_face"], uxds_pair.sizes["n_face"])
    nt.assert_equal(set(uxds_single.data_vars), set(uxds_pair.data_vars))
    assert "ssh" in uxds_single.data_vars


def test_open_dataset_single_combined_xarray_dataset(gridpath):
    """Loads a combined MPAS grid-and-data xarray.Dataset with a single argument."""

    file_path = gridpath("mpas", "QU", "oQU480.231010.nc")

    with xr.open_dataset(file_path) as ds:
        uxds = ux.open_dataset(ds)

    nt.assert_equal(uxds.uxgrid.source_grid_spec, "MPAS")
    nt.assert_equal(uxds.source_datasets, None)
    assert "ssh" in uxds.data_vars


def test_open_dataset_single_argument_rejects_directory_grid(tmp_path):
    """Requires a separate data file for directory-based grids."""

    with pytest.raises(
        ValueError, match="single directory argument is not supported"
    ):
        ux.open_dataset(tmp_path)


def test_open_dataset_single_argument_rejects_invalid_combined_file(datasetpath):
    """Rejects one-file inputs that do not contain recognizable grid metadata."""

    data_path = datasetpath("ugrid", "outCSne30", "outCSne30_var2.nc")

    with pytest.raises(ux.errors.GridInvalidError, match="Failed to parse uxgrid information from xarray.Dataset."):
        ux.open_dataset(data_path)


def test_open_mf_dataset(gridpath, datasetpath, mesh_constants):
    """Loads multiple datasets with their grid topology file using
    uxarray's open_dataset call."""

    grid_path = gridpath("ugrid", "outCSne30", "outCSne30.ug")
    dsfiles_mf_ne30 = datasetpath(
        "ugrid",
        "outCSne30",
        ["outCSne30_var2.nc", "outCSne30_vortex.nc"],
    )
    uxds_mf_ne30 = ux.open_mfdataset(grid_path, dsfiles_mf_ne30)

    nt.assert_equal(uxds_mf_ne30.uxgrid.node_lon.size, mesh_constants['NNODES_outCSne30'])
    nt.assert_equal(len(uxds_mf_ne30.uxgrid._ds.data_vars), mesh_constants['DATAVARS_outCSne30'])
    nt.assert_equal(uxds_mf_ne30.source_datasets, str(dsfiles_mf_ne30))

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


def test_open_dataset_with_fallback_chains_both_engine_errors(tmp_path):
    """When both engines fail, the fallback error must be chained onto the
    default engine's error rather than replacing it."""

    not_netcdf = tmp_path / "not_netcdf.nc"
    not_netcdf.write_text("this is not a netcdf file")

    with pytest.raises(Exception) as excinfo:
        _open_dataset_with_fallback(str(not_netcdf))

    assert excinfo.value.__cause__ is not None, (
        "the default engine's error was discarded"
    )


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


def test_open_multigrid_mask_zero_faces(gridpath):
    """Applying masks that deactivate an entire grid should not fail."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")
    mask_file = gridpath("scrip", "oasis", "masks_no_atm.nc")

    grids = ux.open_multigrid(grid_file, mask_filename=mask_file)

    assert grids["ocn"].n_face == 8
    assert grids["atm"].n_face == 0


def test_open_multigrid_missing_grid_error(gridpath):
    """Requesting a missing grid should raise."""
    grid_file = gridpath("scrip", "oasis", "grids.nc")

    with pytest.raises(ValueError, match="Grid 'land' not found"):
        ux.open_multigrid(grid_file, gridnames=["land"])


def test_concat_various_inputs():
    """Ensure concat() requires uxarray objs and raises clear TypeError otherwise,
    and that concat() works in basic cases (exactly two UxDataArrays or two UxDatasets)
    Includes regression test for first and third bugs mentioned in #1642.
    """
    # ensure reasonable crash when providing no inputs
    with pytest.raises(ValueError, match="requires at least one object"):
        ux.concat((), dim='anything')
    # ensure crash when providing no uxarray objects
    with pytest.raises(TypeError, match="expected either all UxDataArray or all UxDataset"):
        ux.concat((7, "not a uxarray object"), dim='anything')
    # ensure crash when providing some non-uxarray objects
    uxds0 = ux.tutorial.open_dataset('quad-hexagon')
    uxds1 = uxds0 + 10
    with pytest.raises(TypeError, match="expected either all UxDataArray or all UxDataset"):
        ux.concat((uxds0, "not a uxarray object", uxds1), dim='new_dim')
    # ensure crash when providing a mix of UxDataArray and UxDataset objects
    uxarr0 = uxds0['t2m']
    with pytest.raises(TypeError, match="expected either all UxDataArray or all UxDataset"):
        ux.concat((uxds0, uxds1, uxarr0), dim='new_dim')
    with pytest.raises(TypeError, match="expected either all UxDataArray or all UxDataset"):
        ux.concat((uxarr0, uxds0), dim='new_dim')
    # ensure concat works in basic cases: two UxDatasets or two UxDataArrays
    ux.concat((uxds0, uxds1), 'new_dim')  # also ensures can provide dim as positional arg.
    uxarr1 = uxds1['t2m']
    assert uxarr0.uxgrid == uxarr1.uxgrid
    result = ux.concat((uxarr0, uxarr1), 'new_dim')
    # ^^ also serves as regression test for third bug in #1642,
    # i.e.: ensures can actually concat UxDataArray objects.
    # include a few quick tests that the result looks reasonable:
    assert isinstance(result, ux.UxDataArray)
    assert result.uxgrid == uxarr0.uxgrid
    assert result.sizes == {**uxarr0.sizes, 'new_dim': 2}

    # regression test for first bug in #1642: using non-uxarray objects with equal uxgrid
    #    should raise error message mentioning uxarray objects, not xarray objects.
    class Foo():
        def __init__(self, uxgrid):
            self.uxgrid = uxgrid
    foo0 = Foo(7)
    foo1 = Foo(7)  # also 7 because: want to use the same uxgrid value in both cases.
    with pytest.raises(TypeError, match="expected either all UxDataArray or all UxDataset"):
        ux.concat([foo0, foo1], dim='anything')


def test_concat_checks_uxgrid():
    """Ensure concat() requires all objects' uxgrids to be equal,
    but not necessarily the same exact object.
    Includes regression test for second bug mentioned in #1642.
    """
    arrA = ux.tutorial.open_dataset("outCSne30-vortex")['psi']
    arrB = arrA.copy()
    assert arrA.uxgrid == arrB.uxgrid
    assert arrA.uxgrid is not arrB.uxgrid
    ux.concat([arrA, arrB], 'new_dim')

    arrC = ux.tutorial.open_dataset("quad-hexagon")['t2m']
    assert arrA.uxgrid != arrC.uxgrid
    with pytest.raises(ux.errors.GridsMismatchError, match=r"got objs\[1\]\.uxgrid != objs\[0\]\.uxgrid"):
        ux.concat([arrA, arrC], dim='new_dim')
    with pytest.raises(ux.errors.GridsMismatchError, match=r"got objs\[2\]\.uxgrid != objs\[0\]\.uxgrid"):
        ux.concat([arrA, arrB, arrC], dim='new_dim')
