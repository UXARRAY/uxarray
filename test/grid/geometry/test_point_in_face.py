import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import INT_FILL_VALUE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.utils import _get_cartesian_face_edge_nodes_array, _get_cartesian_face_edge_nodes
from uxarray.grid.point_in_face import _point_in_face_from_grid


def test_point_inside(gridpath):
    """Test the function `point_in_face`, where the points are all inside the face"""

    grid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))
    grid.normalize_cartesian_coordinates()

    # Loop through each face
    for i in range(grid.n_face):
        # Set the point as the face center of the polygon
        point_xyz = np.array([grid.face_x[i].values, grid.face_y[i].values, grid.face_z[i].values])
        # Assert that the point is in the polygon
        assert _point_in_face_from_grid(point_xyz, grid, i)


def test_point_outside(gridpath):
    """Test the function `point_in_face`, where the point is outside the face"""

    grid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    # Set the point as the face center of a different face than the face tested
    point_xyz = np.array([grid.face_x[1].values, grid.face_y[1].values, grid.face_z[1].values])

    # Assert that the point is not in the face tested
    assert not _point_in_face_from_grid(point_xyz, grid, fidx=0)


def test_point_on_node(gridpath):
    """Test the function `point_in_face`, when the point is on the node of the polygon"""

    grid = ux.open_grid(gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"))

    # Set the point as a node
    point_xyz = np.array([grid.node_x[0].values, grid.node_y[0].values, grid.node_z[0].values])

    # Assert that the point is in the face
    assert _point_in_face_from_grid(point_xyz, grid, fidx=0)


def test_point_inside_close():
    """Test the function `point_in_face`, where the point is inside the face, but very close to the edge"""

    # Create a square
    vertices_lonlat = [[-10.0, 10.0], [-10.0, -10.0], [10.0, -10.0], [10.0, 10.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    # Choose a point just inside the square
    point = np.array(_lonlat_rad_to_xyz(np.deg2rad(0.0), np.deg2rad(-9.8)))

    # Create the grid and face edges
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)

    assert _point_in_face_from_grid(point, grid, fidx=0)


def test_point_outside_close():
    """Test the function `point_in_face`, where the point is outside the face, but very close to the edge"""

    # Create a square
    vertices_lonlat = [[-10.0, 10.0], [-10.0, -10.0], [10.0, -10.0], [10.0, 10.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    # Choose a point just outside the square
    point = np.array(_lonlat_rad_to_xyz(np.deg2rad(0.0), np.deg2rad(-10.2)))

    # Create the grid and face edges
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)

    assert not _point_in_face_from_grid(point, grid, fidx=0)
