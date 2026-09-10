import uxarray as ux
import os
import pytest
import xarray as xr
import numpy as np
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE


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


def test_encode_esmf_ragged_indices_are_usable(tmp_path):
    """Every index written to elementConn names a real node or is the fill value.

    `elementConn` is encoded as int32. Offsetting INT_FILL_VALUE along with the
    valid indices leaves `-2**63 + 1` in the padded slots, which the narrowing
    truncates to `1` -- node 0, indistinguishable from a real vertex to any
    reader. Nothing raises, so check the encoded output rather than a round trip.
    """
    uxgrid = ux.Grid.from_topology(
        node_lon=np.array([0.0, 10.0, 10.0, 0.0, 20.0]),
        node_lat=np.array([0.0, 0.0, 10.0, 10.0, 0.0]),
        face_node_connectivity=np.array([
            [0, 1, 2, 3],
            [1, 4, 2, INT_FILL_VALUE],
            [0, 3, 4, INT_FILL_VALUE],
        ]),
        fill_value=INT_FILL_VALUE,
    )

    # The padded slots are exactly the ones numElementConn reports as unused
    is_padding = np.array([
        [False, False, False, False],
        [False, False, False, True],
        [False, False, False, True],
    ])

    encoded = uxgrid.to_xarray("ESMF")
    assert encoded["elementConn"].attrs["_FillValue"] == -1
    np.testing.assert_array_equal(encoded["elementConn"].values == -1, is_padding)
    np.testing.assert_array_equal(encoded["numElementConn"].values, [4, 3, 3])

    # Check what actually lands on disk: the fill value has to survive int32
    path = tmp_path / "esmf_ragged.nc"
    encoded.to_netcdf(path)
    with xr.open_dataset(path, mask_and_scale=False) as ds:
        on_disk = ds["elementConn"].values

    assert on_disk.dtype == np.int32
    np.testing.assert_array_equal(on_disk == -1, is_padding)

    valid = on_disk[~is_padding]
    assert valid.min() >= 1, "elementConn holds an index below 1"
    assert valid.max() <= uxgrid.n_node, "elementConn indexes a node that does not exist"
