import os
import numpy as np
import pytest

import uxarray as ux


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
