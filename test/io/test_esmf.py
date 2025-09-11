import uxarray as ux
import os
from pathlib import Path
import pytest
import xarray as xr
import numpy as np
from uxarray.constants import ERROR_TOLERANCE


def test_read_esmf(gridpath):
    """Tests the reading of an ESMF grid file and its encoding into the UGRID
    conventions."""
    uxgrid = ux.open_grid(gridpath("esmf", "ne30", "ne30pg3.grid.nc"))

    dims = ['n_node', 'n_face', 'n_max_face_nodes']
    coords = ['node_lon', 'node_lat', 'face_lon', 'face_lat']
    conns = ['face_node_connectivity', 'n_nodes_per_face']

    for dim in dims:
        assert dim in uxgrid._ds.dims

    for coord in coords:
        assert coord in uxgrid._ds

    for conn in conns:
        assert conn in uxgrid._ds

def test_read_esmf_dataset(gridpath, datasetpath):
    """Tests the constructing of a UxDataset from an ESMF Grid and Data
    File."""
    uxds = ux.open_dataset(gridpath("esmf", "ne30", "ne30pg3.grid.nc"),
                           datasetpath("esmf", "ne30", "ne30pg3.data.nc"))

    dims = ['n_node', 'n_face']

    for dim in dims:
        assert dim in uxds.dims

def test_esmf_round_trip_consistency(gridpath):
    """Test round-trip serialization of grid objects through ESMF xarray format.

    Validates that grid objects can be successfully converted to ESMF xarray.Dataset
    format, serialized to disk, and reloaded while maintaining numerical accuracy
    and topological integrity.

    The test verifies:
    - Successful conversion to ESMF xarray format
    - File I/O round-trip consistency
    - Preservation of face-node connectivity (exact)
    - Preservation of node coordinates (within numerical tolerance)

    Raises:
        AssertionError: If any round-trip validation fails
    """
    # Load original grid
    original_grid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # Convert to ESMF xarray format
    esmf_dataset = original_grid.to_xarray("ESMF")

    # Verify dataset structure
    assert isinstance(esmf_dataset, xr.Dataset)
    assert 'nodeCoords' in esmf_dataset
    assert 'elementConn' in esmf_dataset

    # Define output file path
    esmf_filepath = "test_esmf_ne30.nc"

    # Remove existing test file to ensure clean state
    if os.path.exists(esmf_filepath):
        os.remove(esmf_filepath)

    try:
        # Serialize dataset to disk
        esmf_dataset.to_netcdf(esmf_filepath)

        # Reload grid from serialized file
        reloaded_grid = ux.open_grid(esmf_filepath)

        # Validate topological consistency (face-node connectivity)
        # Integer connectivity arrays must be exactly preserved
        np.testing.assert_array_equal(
            original_grid.face_node_connectivity.values,
            reloaded_grid.face_node_connectivity.values,
            err_msg="ESMF face connectivity mismatch"
        )

        # Validate coordinate consistency with numerical tolerance
        # Coordinate transformations and I/O precision may introduce minor differences
        np.testing.assert_allclose(
            original_grid.node_lon.values,
            reloaded_grid.node_lon.values,
            err_msg="ESMF longitude mismatch",
            rtol=ERROR_TOLERANCE
        )
        np.testing.assert_allclose(
            original_grid.node_lat.values,
            reloaded_grid.node_lat.values,
            err_msg="ESMF latitude mismatch",
            rtol=ERROR_TOLERANCE
        )

    finally:
        # Clean up temporary test file
        if os.path.exists(esmf_filepath):
            os.remove(esmf_filepath)
