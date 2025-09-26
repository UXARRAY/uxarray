import os
import numpy as np
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE


def test_normalize_existing_coordinates_non_norm_initial(gridpath):
    from uxarray.grid.validation import _check_normalization
    uxgrid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    uxgrid.node_x.data = 5 * uxgrid.node_x.data
    uxgrid.node_y.data = 5 * uxgrid.node_y.data
    uxgrid.node_z.data = 5 * uxgrid.node_z.data
    assert not _check_normalization(uxgrid)

    uxgrid.normalize_cartesian_coordinates()
    assert _check_normalization(uxgrid)

def test_normalize_existing_coordinates_norm_initial(gridpath):
    """Test normalization of coordinates that are already normalized."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # Store original coordinates
    orig_x = uxgrid.node_x.copy()
    orig_y = uxgrid.node_y.copy()
    orig_z = uxgrid.node_z.copy()

    # Normalize (should be no-op)
    uxgrid.normalize_cartesian_coordinates()

    # Should be unchanged
    np.testing.assert_allclose(uxgrid.node_x, orig_x)
    np.testing.assert_allclose(uxgrid.node_y, orig_y)
    np.testing.assert_allclose(uxgrid.node_z, orig_z)

def test_sphere_radius_mpas_ocean(gridpath):
    """Test sphere radius functionality with MPAS ocean mesh."""
    # Test with MPAS ocean mesh file
    mpas_ocean_file = gridpath("mpas", "QU", "oQU480.231010.nc")
    grid = ux.open_grid(mpas_ocean_file)

    # Check that MPAS sphere radius is preserved (Earth's radius)
    assert np.isclose(grid.sphere_radius, 6371229.0, rtol=1e-10)

    # Test setting a new radius
    new_radius = 1000.0
    grid.sphere_radius = new_radius
    assert np.isclose(grid.sphere_radius, new_radius)

def test_set_lon_range_attrs(gridpath):
    """Test setting longitude range attributes."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # Test setting longitude range
    uxgrid.node_lon.attrs['valid_range'] = [-180.0, 180.0]

    # Should preserve the attribute
    assert 'valid_range' in uxgrid.node_lon.attrs
    assert uxgrid.node_lon.attrs['valid_range'] == [-180.0, 180.0]


def test_grid_ugrid_exodus_roundtrip(gridpath):
    """Test round-trip serialization of grid objects through UGRID and Exodus xarray formats.

    Validates that grid objects can be successfully converted to xarray.Dataset
    objects in both UGRID and Exodus formats, serialized to disk, and reloaded
    while maintaining numerical accuracy and topological integrity.

    The test verifies:
    - Successful conversion to UGRID and Exodus xarray formats
    - File I/O round-trip consistency
    - Preservation of face-node connectivity (exact)
    - Preservation of node coordinates (within numerical tolerance)

    Raises:
        AssertionError: If any round-trip validation fails
    """
    # Load grids
    grid_CSne30 = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))
    grid_RLL1deg = ux.open_grid(gridpath("ugrid", "outRLL1deg", "outRLL1deg.ug"))
    grid_RLL10deg_CSne4 = ux.open_grid(gridpath("ugrid", "ov_RLL10deg_CSne4", "ov_RLL10deg_CSne4.ug"))

    # Convert grids to xarray.Dataset objects in different formats
    ugrid_datasets = {
        'CSne30': grid_CSne30.to_xarray("UGRID"),
        'RLL1deg': grid_RLL1deg.to_xarray("UGRID"),
        'RLL10deg_CSne4': grid_RLL10deg_CSne4.to_xarray("UGRID")
    }

    exodus_datasets = {
        'CSne30': grid_CSne30.to_xarray("Exodus"),
        'RLL1deg': grid_RLL1deg.to_xarray("Exodus"),
        'RLL10deg_CSne4': grid_RLL10deg_CSne4.to_xarray("Exodus")
    }

    # Define test cases with corresponding grid objects
    test_grids = {
        'CSne30': grid_CSne30,
        'RLL1deg': grid_RLL1deg,
        'RLL10deg_CSne4': grid_RLL10deg_CSne4
    }

    # Perform round-trip validation for each grid type
    test_files = []

    for grid_name in test_grids.keys():
        ugrid_dataset = ugrid_datasets[grid_name]
        exodus_dataset = exodus_datasets[grid_name]
        original_grid = test_grids[grid_name]

        # Define output file paths
        ugrid_filepath = f"test_ugrid_{grid_name}.nc"
        exodus_filepath = f"test_exodus_{grid_name}.exo"
        test_files.append(ugrid_filepath)
        test_files.append(exodus_filepath)

        # Serialize datasets to disk
        ugrid_dataset.to_netcdf(ugrid_filepath)
        exodus_dataset.to_netcdf(exodus_filepath)

        # Reload grids from serialized files
        reloaded_ugrid = ux.open_grid(ugrid_filepath)
        reloaded_exodus = ux.open_grid(exodus_filepath)

        # Validate topological consistency (face-node connectivity)
        # Integer connectivity arrays must be exactly preserved
        np.testing.assert_array_equal(
            original_grid.face_node_connectivity.values,
            reloaded_ugrid.face_node_connectivity.values,
            err_msg=f"UGRID face connectivity mismatch for {grid_name}"
        )
        np.testing.assert_array_equal(
            original_grid.face_node_connectivity.values,
            reloaded_exodus.face_node_connectivity.values,
            err_msg=f"Exodus face connectivity mismatch for {grid_name}"
        )

        # Validate coordinate consistency with numerical tolerance
        # Coordinate transformations and I/O precision may introduce minor differences
        np.testing.assert_allclose(
            original_grid.node_lon.values,
            reloaded_ugrid.node_lon.values,
            err_msg=f"UGRID longitude mismatch for {grid_name}",
            rtol=ERROR_TOLERANCE
        )
        np.testing.assert_allclose(
            original_grid.node_lon.values,
            reloaded_exodus.node_lon.values,
            err_msg=f"Exodus longitude mismatch for {grid_name}",
            rtol=ERROR_TOLERANCE
        )
        np.testing.assert_allclose(
            original_grid.node_lat.values,
            reloaded_ugrid.node_lat.values,
            err_msg=f"UGRID latitude mismatch for {grid_name}",
            rtol=ERROR_TOLERANCE
        )
        np.testing.assert_allclose(
            original_grid.node_lat.values,
            reloaded_exodus.node_lat.values,
            err_msg=f"Exodus latitude mismatch for {grid_name}",
            rtol=ERROR_TOLERANCE
        )

    # This might be need for windows "PermissionError: [WinError 32] -- file accessed by another process"
    reloaded_exodus._ds.close()
    reloaded_ugrid._ds.close()
    del reloaded_exodus
    del reloaded_ugrid

    # Clean up temporary test files
    for filepath in test_files:
        if os.path.exists(filepath):
            os.remove(filepath)
