import os
import numpy as np
import numpy.testing as nt
import pytest

from pathlib import Path

import uxarray as ux
from uxarray.grid.coordinates import _normalize_xyz
from uxarray.grid.geometry import _pole_point_inside_polygon_cartesian

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parent


def test_pole_point_inside_polygon_from_vertice_north():
    vertices = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5]]

    for i, vertex in enumerate(vertices):
        float_vertex = [float(coord) for coord in vertex]
        vertices[i] = _normalize_xyz(*float_vertex)

    face_edge_cart = np.array([[vertices[0], vertices[1]],
                               [vertices[1], vertices[2]],
                               [vertices[2], vertices[3]],
                               [vertices[3], vertices[0]]])

    result = _pole_point_inside_polygon_cartesian('North', face_edge_cart)
    assert result, "North pole should be inside the polygon"

    result = _pole_point_inside_polygon_cartesian('South', face_edge_cart)
    assert not result, "South pole should not be inside the polygon"


def test_pole_point_inside_polygon_from_vertice_south():
    vertices = [[0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [0.0, 0.0, -1.0]]

    for i, vertex in enumerate(vertices):
        float_vertex = [float(coord) for coord in vertex]
        vertices[i] = _normalize_xyz(*float_vertex)

    face_edge_cart = np.array([[vertices[0], vertices[1]],
                               [vertices[1], vertices[2]],
                               [vertices[2], vertices[0]]])

    result = _pole_point_inside_polygon_cartesian('North', face_edge_cart)
    assert not result, "North pole should not be inside the polygon"

    result = _pole_point_inside_polygon_cartesian('South', face_edge_cart)
    assert result, "South pole should be inside the polygon"


def test_pole_point_inside_polygon_from_vertice_pole():
    vertices = [[0, 0, 1], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5]]

    for i, vertex in enumerate(vertices):
        float_vertex = [float(coord) for coord in vertex]
        vertices[i] = _normalize_xyz(*float_vertex)

    face_edge_cart = np.array([[vertices[0], vertices[1]],
                               [vertices[1], vertices[2]],
                               [vertices[2], vertices[3]],
                               [vertices[3], vertices[0]]])

    result = _pole_point_inside_polygon_cartesian('North', face_edge_cart)
    assert result, "North pole should be inside the polygon"

    result = _pole_point_inside_polygon_cartesian('South', face_edge_cart)
    assert not result, "South pole should not be inside the polygon"


def test_pole_point_inside_polygon_from_vertice_cross():
    vertices = [[0.6, -0.3, 0.5], [0.2, 0.2, -0.2], [-0.5, 0.1, -0.2],
                [-0.1, -0.2, 0.2]]

    for i, vertex in enumerate(vertices):
        float_vertex = [float(coord) for coord in vertex]
        vertices[i] = _normalize_xyz(*float_vertex)

    face_edge_cart = np.array([[vertices[0], vertices[1]],
                               [vertices[1], vertices[2]],
                               [vertices[2], vertices[3]],
                               [vertices[3], vertices[0]]])

    result = _pole_point_inside_polygon_cartesian('North', face_edge_cart)
    assert result, "North pole should be inside the polygon"


def test_face_at_pole():
    """Test for a face that contains a pole."""
    # Create a face that contains the north pole
    verts = [[[0, 80], [120, 80], [240, 80]]]
    uxgrid = ux.open_grid(verts, latlon=True)

    # Test that the grid is created successfully
    assert uxgrid.n_face == 1
    assert uxgrid.n_node == 3
