import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import INT_FILL_VALUE, ERROR_TOLERANCE
from uxarray.grid.coordinates import _populate_node_latlon


# Test data for grid initialization
f0_deg = [[120, -20], [130, -10], [120, 0], [105, 0], [95, -10], [105, -20]]
f1_deg = [[120, 0], [120, 10], [115, 0],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]
f2_deg = [[115, 0], [120, 10], [100, 10], [105, 0],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE],
          [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]]


def test_grid_init_verts():
    """Test grid initialization from vertices."""
    # Test with simple triangle
    vertices = [
        [0.0, 90.0],
        [0.0, 0.0],
        [90.0, 0.0]
    ]

    uxgrid = ux.Grid.from_face_vertices(vertices, latlon=True)

    # Basic validation
    assert uxgrid.n_face == 1
    assert uxgrid.n_node == 3
    assert uxgrid.n_edge == 3

def test_grid_init_verts_different_input_datatype():
    """Test grid initialization with different input data types."""
    # Test with numpy array
    vertices_np = np.array([
        [0.0, 90.0],
        [0.0, 0.0],
        [90.0, 0.0]
    ])

    uxgrid_np = ux.Grid.from_face_vertices(vertices_np, latlon=True)

    # Test with list
    vertices_list = [
        [0.0, 90.0],
        [0.0, 0.0],
        [90.0, 0.0]
    ]

    uxgrid_list = ux.Grid.from_face_vertices(vertices_list, latlon=True)

    # Should produce equivalent grids
    assert uxgrid_np.n_face == uxgrid_list.n_face
    assert uxgrid_np.n_node == uxgrid_list.n_node

def test_grid_init_verts_fill_values():
    """Test grid initialization with fill values."""
    # Test with mixed face types (triangle and quad with fill values)
    face_vertices = [
        [[0, 0], [1, 0], [0.5, 1], [ux.INT_FILL_VALUE, ux.INT_FILL_VALUE]],  # Triangle with fill
        [[1, 0], [2, 0], [2, 1], [1, 1]]  # Quad
    ]

    uxgrid = ux.Grid.from_face_vertices(face_vertices, latlon=True)

    # Should handle fill values correctly
    assert uxgrid.n_face == 2
    assert uxgrid.n_node > 0

def test_grid_properties(gridpath):
    """Test grid properties."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # Test basic properties
    assert uxgrid.n_face > 0
    assert uxgrid.n_node > 0
    assert uxgrid.n_edge > 0

    # Test coordinate properties
    assert hasattr(uxgrid, 'node_lon')
    assert hasattr(uxgrid, 'node_lat')
    assert len(uxgrid.node_lon) == uxgrid.n_node
    assert len(uxgrid.node_lat) == uxgrid.n_node

    # Test connectivity properties
    assert hasattr(uxgrid, 'face_node_connectivity')
    assert uxgrid.face_node_connectivity.shape[0] == uxgrid.n_face

def test_read_shpfile(test_data_dir):
    """Reads a shape file and write ugrid file."""
    shp_filename = test_data_dir / "shp" / "grid_fire.shp"
    with pytest.raises(ValueError):
        grid_shp = ux.open_grid(shp_filename)

def test_populate_coordinates_populate_cartesian_xyz_coord():
    """Test population of Cartesian XYZ coordinates."""
    # Create simple grid
    vertices = [
        [0.0, 90.0],
        [0.0, 0.0],
        [90.0, 0.0]
    ]

    uxgrid = ux.Grid.from_face_vertices(vertices, latlon=True)

    # Should have Cartesian coordinates
    assert hasattr(uxgrid, 'node_x')
    assert hasattr(uxgrid, 'node_y')
    assert hasattr(uxgrid, 'node_z')

    # Check coordinate ranges
    assert np.all(uxgrid.node_x**2 + uxgrid.node_y**2 + uxgrid.node_z**2 <= 1.0 + ERROR_TOLERANCE)

def test_populate_coordinates_populate_lonlat_coord():
    lon_deg = [
        45.0001052295749, 45.0001052295749, 360 - 45.0001052295749,
                                            360 - 45.0001052295749
    ]
    lat_deg = [
        35.2655522903022, -35.2655522903022, 35.2655522903022,
        -35.2655522903022
    ]
    cart_x = [
        0.577340924821405, 0.577340924821405, 0.577340924821405,
        0.577340924821405
    ]
    cart_y = [
        0.577343045516932, 0.577343045516932, -0.577343045516932,
        -0.577343045516932
    ]
    cart_z = [
        0.577366836872017, -0.577366836872017, 0.577366836872017,
        -0.577366836872017
    ]

    verts_cart = np.stack((cart_x, cart_y, cart_z), axis=1)
    vgrid = ux.open_grid(verts_cart, latlon=False)
    _populate_node_latlon(vgrid)
    lon_deg, lat_deg = zip(*reversed(list(zip(lon_deg, lat_deg))))
    for i in range(0, vgrid.n_node):
        nt.assert_almost_equal(vgrid._ds["node_lon"].values[i], lon_deg[i], decimal=12)
        nt.assert_almost_equal(vgrid._ds["node_lat"].values[i], lat_deg[i], decimal=12)
