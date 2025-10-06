import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import INT_FILL_VALUE, ERROR_TOLERANCE


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


def test_read_shpfile(test_data_dir):
    """Reads a shape file and write ugrid file."""
    shp_filename = test_data_dir / "shp" / "grid_fire.shp"
    with pytest.raises((ValueError, FileNotFoundError, OSError)):
        grid_shp = ux.open_grid(shp_filename)


def test_class_methods_from_face_vertices():
    """Test class methods for creating grids from face vertices."""
    single_face_latlon = [(0.0, 90.0), (-180, 0.0), (0.0, -90)]
    uxgrid = ux.Grid.from_face_vertices(single_face_latlon, latlon=True)

    multi_face_latlon = [[(0.0, 90.0), (-180, 0.0), (0.0, -90)],
                         [(0.0, 90.0), (180, 0.0), (0.0, -90)]]
    uxgrid = ux.Grid.from_face_vertices(multi_face_latlon, latlon=True)

    single_face_cart = [(0.0,)]


def test_from_topology():
    """Test grid creation from topology."""
    node_lon = np.array([-20.0, 0.0, 20.0, -20, -40])
    node_lat = np.array([-10.0, 10.0, -10.0, 10, -10])
    face_node_connectivity = np.array([[0, 1, 2, -1], [0, 1, 3, 4]])

    uxgrid = ux.Grid.from_topology(
        node_lon=node_lon,
        node_lat=node_lat,
        face_node_connectivity=face_node_connectivity,
        fill_value=-1,
    )
