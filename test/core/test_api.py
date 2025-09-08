import os
from pathlib import Path
import numpy.testing as nt
import uxarray as ux
import numpy as np
import pytest

# Import centralized paths
import sys
sys.path.append(str(Path(__file__).parent.parent))
from paths import *
try:
    import constants
except ImportError:
    from . import constants


geoflow_data_v1 = GEOFLOW_V1
geoflow_data_v2 = GEOFLOW_V2
geoflow_data_v3 = GEOFLOW_V3

dsfiles_mf_ne30 = str(MESHFILES_PATH) + "/ugrid/outCSne30/outCSne30_*.nc"

def test_open_geoflow_dataset():
    """Loads a single dataset with its grid topology file using uxarray's
    open_dataset call."""

    # Paths to Data Variable files
    data_paths = [
        geoflow_data_v1, geoflow_data_v2, geoflow_data_v3
    ]

    uxds_v1 = ux.open_dataset(GEOFLOW_GRID, data_paths[0])

    # Ideally uxds_v1.uxgrid should NOT be None
    nt.assert_equal(uxds_v1.uxgrid is not None, True)

def test_open_dataset():
    """Loads a single dataset with its grid topology file using uxarray's
    open_dataset call."""

    uxds_var2_ne30 = ux.open_dataset(OUTCSNE30_GRID, OUTCSNE30_VAR2)

    nt.assert_equal(uxds_var2_ne30.uxgrid.node_lon.size, constants.NNODES_outCSne30)
    nt.assert_equal(len(uxds_var2_ne30.uxgrid._ds.data_vars), constants.DATAVARS_outCSne30)
    nt.assert_equal(uxds_var2_ne30.source_datasets, str(OUTCSNE30_VAR2))

def test_open_mf_dataset():
    """Loads multiple datasets with their grid topology file using
    uxarray's open_dataset call."""

    uxds_mf_ne30 = ux.open_mfdataset(OUTCSNE30_GRID, dsfiles_mf_ne30)

    nt.assert_equal(uxds_mf_ne30.uxgrid.node_lon.size, constants.NNODES_outCSne30)
    nt.assert_equal(len(uxds_mf_ne30.uxgrid._ds.data_vars), constants.DATAVARS_outCSne30)
    nt.assert_equal(uxds_mf_ne30.source_datasets, dsfiles_mf_ne30)

def test_open_grid():
    """Loads only a grid topology file using uxarray's open_grid call."""
    uxgrid = ux.open_grid(GEOFLOW_GRID)

    nt.assert_almost_equal(uxgrid.calculate_total_face_area(), constants.MESH30_AREA, decimal=3)

def test_copy_dataset():
    """Loads a single dataset with its grid topology file using uxarray's
    open_dataset call and make a copy of the object."""

    uxds_var2_ne30 = ux.open_dataset(OUTCSNE30_GRID, OUTCSNE30_VAR2)

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

def test_copy_dataarray():
    """Loads an unstructured grid and data using uxarray's open_dataset
    call and make a copy of the dataarray object."""

    # Paths to Data Variable files
    data_paths = [
        geoflow_data_v1, geoflow_data_v2, geoflow_data_v3
    ]

    uxds_v1 = ux.open_dataset(GEOFLOW_GRID, data_paths[0])

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

def test_open_dataset_grid_kwargs():
    """Drops ``Mesh2_face_nodes`` from the inputted grid file using
    ``grid_kwargs``"""

    with pytest.raises(ValueError):
        # attempt to open a dataset after dropping face nodes should raise a KeyError
        uxds = ux.open_dataset(
            OUTCSNE30_GRID,
            OUTCSNE30_VAR2,
            grid_kwargs={"drop_variables": "Mesh2_face_nodes"}
        )
