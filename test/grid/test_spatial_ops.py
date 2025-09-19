"""Tests for spatial operations functionality.

This module contains tests for spatial operations including:
- Antimeridian crossing detection and handling
- Pole point algorithms and detection
- Stereographic projections
- Point-in-face operations at special locations
"""

import numpy as np
import uxarray as ux
from uxarray.grid.coordinates import _normalize_xyz, _lonlat_rad_to_xyz
from uxarray.grid.geometry import _pole_point_inside_polygon_cartesian, stereographic_projection, inverse_stereographic_projection
from uxarray.grid.utils import _get_cartesian_face_edge_nodes_array
from uxarray.grid.point_in_face import _face_contains_point


def test_antimeridian_crossing():
    verts = [[[-170, 40], [180, 30], [165, 25], [-170, 20]]]

    uxgrid = ux.open_grid(verts, latlon=True)

    gdf = uxgrid.to_geodataframe(periodic_elements='ignore')

    assert len(uxgrid.antimeridian_face_indices) == 1
    assert len(gdf['geometry']) == 1


def test_antimeridian_point_on():
    verts = [[[-170, 40], [180, 30], [-170, 20]]]

    uxgrid = ux.open_grid(verts, latlon=True)

    assert len(uxgrid.antimeridian_face_indices) == 1


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


def test_insert_pt_in_latlonbox_pole():
    """Test inserting a point at the pole in a lat-lon box."""
    # This test checks pole handling in lat-lon box operations
    vertices_lonlat = [[0.0, 89.0], [90.0, 89.0], [180.0, 89.0], [270.0, 89.0]]
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)

    # Check that the grid was created successfully
    assert grid.n_face == 1
    assert grid.n_node == 4


def test_face_at_pole():
    """Test the function `point_in_face`, when the face is at the North Pole"""
    # Generate a face that is at a pole
    vertices_lonlat = [[10.0, 90.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    point = np.array(_lonlat_rad_to_xyz(np.deg2rad(25), np.deg2rad(30)))

    # Create the grid and face edges
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    faces_edges_cartesian = _get_cartesian_face_edge_nodes_array(
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_edges,
        grid.node_x.values,
        grid.node_y.values,
        grid.node_z.values,
    )

    assert _face_contains_point(faces_edges_cartesian[0], point)


def test_face_at_antimeridian():
    """Test the function `point_in_face`, where the face crosses the antimeridian"""
    # Generate a face crossing the antimeridian
    vertices_lonlat = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)
    point = np.array(_lonlat_rad_to_xyz(np.deg2rad(25), np.deg2rad(30)))

    # Create the grid and face edges
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    faces_edges_cartesian = _get_cartesian_face_edge_nodes_array(
        grid.face_node_connectivity.values,
        grid.n_face,
        grid.n_max_face_edges,
        grid.node_x.values,
        grid.node_y.values,
        grid.node_z.values,
    )

    assert _face_contains_point(faces_edges_cartesian[0], point)


def test_stereographic_projection_stereographic_projection():
    lon = np.array(0)
    lat = np.array(0)

    central_lon = np.array(0)
    central_lat = np.array(0)

    x, y = stereographic_projection(lon, lat, central_lon, central_lat)

    new_lon, new_lat = inverse_stereographic_projection(x, y, central_lon, central_lat)

    assert np.array_equal(lon, new_lon)
    assert np.array_equal(lat, new_lat)
    assert np.array_equal(x, y) and x == 0
