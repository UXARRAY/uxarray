import os
import numpy as np
import numpy.testing as nt
import xarray as xr
import pytest

from pathlib import Path

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
import uxarray.utils.computing as ac_utils
from uxarray.grid.coordinates import _populate_node_latlon, _lonlat_rad_to_xyz, _normalize_xyz, _xyz_to_lonlat_rad, \
    _xyz_to_lonlat_deg, _xyz_to_lonlat_rad_scalar
from uxarray.grid.arcs import extreme_gca_latitude, extreme_gca_z
from uxarray.grid.utils import _get_cartesian_faces_edge_nodes, _get_lonlat_rad_faces_edge_nodes

from uxarray.grid.geometry import _populate_face_latlon_bound, _populate_bounds, pole_point_inside_polygon, \
    stereographic_projection, inverse_stereographic_projection, point_in_face, haversine_distance

from sklearn.metrics.pairwise import haversine_distances

current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_CSne8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
datafile_CSne30 = current_path / "meshfiles" / "ugrid" / "outCSne30" / "outCSne30_vortex.nc"

gridfile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
datafile_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "v1.nc"

grid_files = [gridfile_CSne8, gridfile_geoflow]
data_files = [datafile_CSne30, datafile_geoflow]

grid_quad_hex = current_path / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"
grid_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
grid_mpas = current_path / "meshfiles" / "mpas" / "QU" / "oQU480.231010.nc"

grid_mpas_2 = current_path / "meshfiles" / "mpas" / "QU" / "mesh.QU.1920km.151026.nc"

# List of grid files to test
grid_files_latlonBound = [grid_quad_hex, grid_geoflow, gridfile_CSne8, grid_mpas]


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


def test_linecollection_execution():
    uxgrid = ux.open_grid(gridfile_CSne8)
    lines = uxgrid.to_linecollection()


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

    x = face_edge_cart[:, :, 0]
    y = face_edge_cart[:, :, 1]
    z = face_edge_cart[:, :, 2]

    lon, lat = _xyz_to_lonlat_rad(x, y, z)

    face_edges_lonlat = np.stack((lon, lat), axis=2)

    result = pole_point_inside_polygon(1, face_edge_cart,face_edges_lonlat )
    assert result, "North pole should be inside the polygon"

    result = pole_point_inside_polygon(-1, face_edge_cart,face_edges_lonlat)
    assert not result, "South pole should not be inside the polygon"


def test_pole_point_inside_polygon_from_vertice_south():
    vertices = [[0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [0.0, 0.0, -1.0]]

    for i, vertex in enumerate(vertices):
        float_vertex = [float(coord) for coord in vertex]
        vertices[i] = _normalize_xyz(*float_vertex)

    face_edge_cart = np.array([[vertices[0], vertices[1]],
                               [vertices[1], vertices[2]],
                               [vertices[2], vertices[0]]])

    x = face_edge_cart[:, :, 0]
    y = face_edge_cart[:, :, 1]
    z = face_edge_cart[:, :, 2]

    lon, lat = _xyz_to_lonlat_rad(x, y, z)

    face_edges_lonlat = np.stack((lon, lat), axis=2)


    result = pole_point_inside_polygon(1, face_edge_cart, face_edges_lonlat)
    assert not result, "North pole should not be inside the polygon"

    result = pole_point_inside_polygon(-1, face_edge_cart, face_edges_lonlat)
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

    x = face_edge_cart[:, :, 0]
    y = face_edge_cart[:, :, 1]
    z = face_edge_cart[:, :, 2]

    lon, lat = _xyz_to_lonlat_rad(x, y, z)

    face_edges_lonlat = np.stack((lon, lat), axis=2)

    result = pole_point_inside_polygon(1, face_edge_cart, face_edges_lonlat)
    assert result, "North pole should be inside the polygon"

    result = pole_point_inside_polygon(-1, face_edge_cart, face_edges_lonlat)
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

    x = face_edge_cart[:, :, 0]
    y = face_edge_cart[:, :, 1]
    z = face_edge_cart[:, :, 2]

    lon, lat = _xyz_to_lonlat_rad(x, y, z)

    face_edges_lonlat = np.stack((lon, lat), axis=2)

    result = pole_point_inside_polygon(1, face_edge_cart,face_edges_lonlat)
    assert result, "North pole should be inside the polygon"


def _max_latitude_rad_iterative(gca_cart):
    """Calculate the maximum latitude of a great circle arc defined by two
    points.

    Parameters
    ----------
    gca_cart : numpy.ndarray
        An array containing two 3D vectors that define a great circle arc.

    Returns
    -------
    float
        The maximum latitude of the great circle arc in radians.

    Raises
    ------
    ValueError
        If the input vectors are not valid 2-element lists or arrays.

    Notes
    -----
    The method divides the great circle arc into subsections, iteratively refining the subsection of interest
    until the maximum latitude is found within a specified tolerance.
    """

    # Convert input vectors to radians and Cartesian coordinates

    v1_cart, v2_cart = gca_cart
    b_lonlat = _xyz_to_lonlat_rad(*v1_cart.tolist())
    c_lonlat = _xyz_to_lonlat_rad(*v2_cart.tolist())

    # Initialize variables for the iterative process
    v_temp = ac_utils.cross_fma(v1_cart, v2_cart)
    v0 = ac_utils.cross_fma(v_temp, v1_cart)
    v0 = _normalize_xyz(*v0.tolist())
    max_section = [v1_cart, v2_cart]

    # Iteratively find the maximum latitude
    while np.abs(b_lonlat[1] - c_lonlat[1]) >= ERROR_TOLERANCE or np.abs(
            b_lonlat[0] - c_lonlat[0]) >= ERROR_TOLERANCE:
        max_lat = -np.pi
        v_b, v_c = max_section
        angle_v1_v2_rad = ux.grid.arcs._angle_of_2_vectors(v_b, v_c)
        v0 = ac_utils.cross_fma(v_temp, v_b)
        v0 = _normalize_xyz(*v0.tolist())
        avg_angle_rad = angle_v1_v2_rad / 10.0

        for i in range(10):
            angle_rad_prev = avg_angle_rad * i
            angle_rad_next = angle_rad_prev + avg_angle_rad if i < 9 else angle_v1_v2_rad
            w1_new = np.cos(angle_rad_prev) * v_b + np.sin(
                angle_rad_prev) * np.array(v0)
            w2_new = np.cos(angle_rad_next) * v_b + np.sin(
                angle_rad_next) * np.array(v0)
            w1_lonlat = _xyz_to_lonlat_rad(
                *w1_new.tolist())
            w2_lonlat = _xyz_to_lonlat_rad(
                *w2_new.tolist())

            w1_lonlat = np.asarray(w1_lonlat)
            w2_lonlat = np.asarray(w2_lonlat)

            # Adjust latitude boundaries to avoid error accumulation
            if i == 0:
                w1_lonlat[1] = b_lonlat[1]
            elif i >= 9:
                w2_lonlat[1] = c_lonlat[1]

            # Update maximum latitude and section if needed
            max_lat = max(max_lat, w1_lonlat[1], w2_lonlat[1])
            if np.abs(w2_lonlat[1] -
                      w1_lonlat[1]) <= ERROR_TOLERANCE or w1_lonlat[
                1] == max_lat == w2_lonlat[1]:
                max_section = [w1_new, w2_new]
                break
            if np.abs(max_lat - w1_lonlat[1]) <= ERROR_TOLERANCE:
                max_section = [w1_new, w2_new] if i != 0 else [v_b, w2_new]
            elif np.abs(max_lat - w2_lonlat[1]) <= ERROR_TOLERANCE:
                max_section = [w1_new, w2_new] if i != 9 else [w1_new, v_c]

        # Update longitude and latitude for the next iteration
        b_lonlat = _xyz_to_lonlat_rad(
            *max_section[0].tolist())
        c_lonlat = _xyz_to_lonlat_rad(
            *max_section[1].tolist())

    return np.average([b_lonlat[1], c_lonlat[1]])


def _min_latitude_rad_iterative(gca_cart):
    """Calculate the minimum latitude of a great circle arc defined by two
    points.

    Parameters
    ----------
    gca_cart : numpy.ndarray
        An array containing two 3D vectors that define a great circle arc.

    Returns
    -------
    float
        The minimum latitude of the great circle arc in radians.

    Raises
    ------
    ValueError
        If the input vectors are not valid 2-element lists or arrays.

    Notes
    -----
    The method divides the great circle arc into subsections, iteratively refining the subsection of interest
    until the minimum latitude is found within a specified tolerance.
    """

    # Convert input vectors to radians and Cartesian coordinates
    v1_cart, v2_cart = gca_cart
    b_lonlat = _xyz_to_lonlat_rad(*v1_cart.tolist())
    c_lonlat = _xyz_to_lonlat_rad(*v2_cart.tolist())

    # Initialize variables for the iterative process
    v_temp = ac_utils.cross_fma(v1_cart, v2_cart)
    v0 = ac_utils.cross_fma(v_temp, v1_cart)
    v0 = np.array(_normalize_xyz(*v0.tolist()))
    min_section = [v1_cart, v2_cart]

    # Iteratively find the minimum latitude
    while np.abs(b_lonlat[1] - c_lonlat[1]) >= ERROR_TOLERANCE or np.abs(
            b_lonlat[0] - c_lonlat[0]) >= ERROR_TOLERANCE:
        min_lat = np.pi
        v_b, v_c = min_section
        angle_v1_v2_rad = ux.grid.arcs._angle_of_2_vectors(v_b, v_c)
        v0 = ac_utils.cross_fma(v_temp, v_b)
        v0 = np.array(_normalize_xyz(*v0.tolist()))
        avg_angle_rad = angle_v1_v2_rad / 10.0

        for i in range(10):
            angle_rad_prev = avg_angle_rad * i
            angle_rad_next = angle_rad_prev + avg_angle_rad if i < 9 else angle_v1_v2_rad
            w1_new = np.cos(angle_rad_prev) * v_b + np.sin(
                angle_rad_prev) * v0
            w2_new = np.cos(angle_rad_next) * v_b + np.sin(
                angle_rad_next) * v0
            w1_lonlat = _xyz_to_lonlat_rad(
                *w1_new.tolist())
            w2_lonlat = _xyz_to_lonlat_rad(
                *w2_new.tolist())

            w1_lonlat = np.asarray(w1_lonlat)
            w2_lonlat = np.asarray(w2_lonlat)

            # Adjust latitude boundaries to avoid error accumulation
            if i == 0:
                w1_lonlat[1] = b_lonlat[1]
            elif i >= 9:
                w2_lonlat[1] = c_lonlat[1]

            # Update minimum latitude and section if needed
            min_lat = min(min_lat, w1_lonlat[1], w2_lonlat[1])
            if np.abs(w2_lonlat[1] -
                      w1_lonlat[1]) <= ERROR_TOLERANCE or w1_lonlat[
                1] == min_lat == w2_lonlat[1]:
                min_section = [w1_new, w2_new]
                break
            if np.abs(min_lat - w1_lonlat[1]) <= ERROR_TOLERANCE:
                min_section = [w1_new, w2_new] if i != 0 else [v_b, w2_new]
            elif np.abs(min_lat - w2_lonlat[1]) <= ERROR_TOLERANCE:
                min_section = [w1_new, w2_new] if i != 9 else [w1_new, v_c]

        # Update longitude and latitude for the next iteration
        b_lonlat = _xyz_to_lonlat_rad(
            *min_section[0].tolist())
        c_lonlat = _xyz_to_lonlat_rad(
            *min_section[1].tolist())

    return np.average([b_lonlat[1], c_lonlat[1]])


def test_extreme_gca_latitude_max():
    gca_cart = np.array([
        _normalize_xyz(*[0.5, 0.5, 0.5]),
        _normalize_xyz(*[-0.5, 0.5, 0.5])
    ])

    max_latitude = extreme_gca_z(gca_cart, 'max')
    expected_max_latitude = np.cos(_max_latitude_rad_iterative(gca_cart))
    assert np.isclose(max_latitude, expected_max_latitude, atol=ERROR_TOLERANCE)

    gca_cart = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    max_latitude = extreme_gca_z(gca_cart, 'max')
    expected_max_latitude = 1.0
    assert np.isclose(max_latitude, expected_max_latitude, atol=ERROR_TOLERANCE)


def test_extreme_gca_latitude_max_short():
    # Define a great circle arc in 3D space that has a small span
    gca_cart = np.array([[0.65465367, -0.37796447, -0.65465367], [0.6652466, -0.33896007, -0.6652466]])

    # Calculate the maximum latitude
    max_latitude = np.asin(extreme_gca_z(gca_cart, 'max'))

    # Check if the maximum latitude is correct
    expected_max_latitude = _max_latitude_rad_iterative(gca_cart)
    assert np.isclose(max_latitude,
                      expected_max_latitude,
                      atol=ERROR_TOLERANCE)


def test_extreme_gca_latitude_min():
    gca_cart = np.array([
        _normalize_xyz(*[0.5, 0.5, -0.5]),
        _normalize_xyz(*[-0.5, 0.5, -0.5])
    ])

    min_latitude = np.asin(extreme_gca_z(gca_cart, 'min'))
    expected_min_latitude = _min_latitude_rad_iterative(gca_cart)
    assert np.isclose(min_latitude, expected_min_latitude, atol=ERROR_TOLERANCE)

    gca_cart = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    min_latitude = np.asin(extreme_gca_z(gca_cart, 'min'))
    expected_min_latitude = -np.pi / 2
    assert np.isclose(min_latitude, expected_min_latitude, atol=ERROR_TOLERANCE)


def test_get_latlonbox_width():
    gca_latlon = np.array([[0.0, 0.0], [0.0, 3.0]])
    width = ux.grid.geometry._get_latlonbox_width(gca_latlon)
    assert width == 3.0

    gca_latlon = np.array([[0.0, 0.0], [2 * np.pi - 1.0, 1.0]])
    width = ux.grid.geometry._get_latlonbox_width(gca_latlon)
    assert width == 2.0


def test_insert_pt_in_latlonbox_non_periodic():
    old_box = np.array([[0.1, 0.2], [0.3, 0.4]])  # Radians
    new_pt = np.array([0.15, 0.35])
    expected = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = ux.grid.geometry.insert_pt_in_latlonbox(old_box, new_pt, False)
    np.testing.assert_array_equal(result, expected)


def test_insert_pt_in_latlonbox_periodic():
    old_box = np.array([[0.1, 0.2], [6.0, 0.1]])  # Radians, periodic
    new_pt = np.array([0.15, 6.2])
    expected = np.array([[0.1, 0.2], [6.0, 0.1]])
    result = ux.grid.geometry.insert_pt_in_latlonbox(old_box, new_pt, True)
    np.testing.assert_array_equal(result, expected)


def test_insert_pt_in_latlonbox_pole():
    old_box = np.array([[0.1, 0.2], [0.3, 0.4]])
    new_pt = np.array([np.pi / 2, np.nan])  # Pole point
    expected = np.array([[0.1, np.pi / 2], [0.3, 0.4]])
    result = ux.grid.geometry.insert_pt_in_latlonbox(old_box, new_pt)
    np.testing.assert_array_equal(result, expected)


def test_insert_pt_in_empty_state():
    old_box = np.array([[np.nan, np.nan],
                        [np.nan, np.nan]])  # Empty state
    new_pt = np.array([0.15, 0.35])
    expected = np.array([[0.15, 0.15], [0.35, 0.35]])
    result = ux.grid.geometry.insert_pt_in_latlonbox(old_box, new_pt)
    np.testing.assert_array_equal(result, expected)


def _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_gca(face_nodes_ind, face_edges_ind, edge_nodes_grid,
                                                                     node_x, node_y, node_z):
    """Construct an array to hold the edge Cartesian coordinates connectivity for a face in a grid."""
    mask = face_edges_ind != INT_FILL_VALUE
    valid_edges = face_edges_ind[mask]
    face_edges = edge_nodes_grid[valid_edges]

    face_edges[0] = [face_nodes_ind[0], face_nodes_ind[1]]

    for idx in range(1, len(face_edges)):
        if face_edges[idx][0] != face_edges[idx - 1][1]:
            face_edges[idx] = face_edges[idx][::-1]

    cartesian_coordinates = np.array(
        [
            [[node_x[node], node_y[node], node_z[node]] for node in edge]
            for edge in face_edges
        ]
    )

    return cartesian_coordinates


def _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_gca(face_nodes_ind, face_edges_ind, edge_nodes_grid,
                                                                      node_lon, node_lat):
    """Construct an array to hold the edge lat lon in radian connectivity for a face in a grid."""
    mask = face_edges_ind != INT_FILL_VALUE
    valid_edges = face_edges_ind[mask]
    face_edges = edge_nodes_grid[valid_edges]

    face_edges[0] = [face_nodes_ind[0], face_nodes_ind[1]]

    for idx in range(1, len(face_edges)):
        if face_edges[idx][0] != face_edges[idx - 1][1]:
            face_edges[idx] = face_edges[idx][::-1]

    lonlat_coordinates = np.array(
        [
            [
                [
                    np.mod(np.deg2rad(node_lon[node]), 2 * np.pi),
                    np.deg2rad(node_lat[node]),
                ]
                for node in edge
            ]
            for edge in face_edges
        ]
    )

    return lonlat_coordinates


def test_populate_bounds_normal_latlon_bounds_gca():
    vertices_lonlat = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
    lat_max = max(np.deg2rad(60.0),
                  np.asin(extreme_gca_z(np.array([vertices_cart[0], vertices_cart[3]]), extreme_type="max")))
    lat_min = min(np.deg2rad(10.0),
                  np.asin(extreme_gca_z(np.array([vertices_cart[1], vertices_cart[2]]), extreme_type="min")))
    lon_min = np.deg2rad(10.0)
    lon_max = np.deg2rad(50.0)
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_gca(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_gca(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_antimeridian_latlon_bounds_gca():
    vertices_lonlat = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
    lat_max = max(np.deg2rad(60.0),
                  np.asin(extreme_gca_z(np.array([vertices_cart[0], vertices_cart[3]]), extreme_type="max")))
    lat_min = min(np.deg2rad(10.0),
                  np.asin(extreme_gca_z(np.array([vertices_cart[1], vertices_cart[2]]), extreme_type="min")))
    lon_min = np.deg2rad(350.0)
    lon_max = np.deg2rad(50.0)
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_gca(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_gca(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_equator_latlon_bounds_gca():
    face_edges_cart = np.array([
        [[0.99726469, -0.05226443, -0.05226443], [0.99862953, 0.0, -0.05233596]],
        [[0.99862953, 0.0, -0.05233596], [1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.99862953, -0.05233596, 0.0]],
        [[0.99862953, -0.05233596, 0.0], [0.99726469, -0.05226443, -0.05226443]]
    ])
    face_edges_lonlat = np.array(
        [[_xyz_to_lonlat_rad(*edge[0]), _xyz_to_lonlat_rad(*edge[1])] for edge in face_edges_cart])

    bounds = _populate_face_latlon_bound(face_edges_cart, face_edges_lonlat)
    expected_bounds = np.array([[-0.05235988, 0], [6.23082543, 0]])
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_south_sphere_latlon_bounds_gca():
    face_edges_cart = np.array([
        [[-1.04386773e-01, -5.20500333e-02, -9.93173799e-01], [-1.04528463e-01, -1.28010448e-17, -9.94521895e-01]],
        [[-1.04528463e-01, -1.28010448e-17, -9.94521895e-01], [-5.23359562e-02, -6.40930613e-18, -9.98629535e-01]],
        [[-5.23359562e-02, -6.40930613e-18, -9.98629535e-01], [-5.22644277e-02, -5.22644277e-02, -9.97264689e-01]],
        [[-5.22644277e-02, -5.22644277e-02, -9.97264689e-01], [-1.04386773e-01, -5.20500333e-02, -9.93173799e-01]]
    ])

    face_edges_lonlat = np.array(
        [[_xyz_to_lonlat_rad(*edge[0]), _xyz_to_lonlat_rad(*edge[1])] for edge in face_edges_cart])

    bounds = _populate_face_latlon_bound(face_edges_cart, face_edges_lonlat)
    expected_bounds = np.array([[-1.51843645, -1.45388627], [3.14159265, 3.92699082]])
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_near_pole_latlon_bounds_gca():
    face_edges_cart = np.array([
        [[3.58367950e-01, 0.00000000e+00, -9.33580426e-01], [3.57939780e-01, 4.88684203e-02, -9.32465008e-01]],
        [[3.57939780e-01, 4.88684203e-02, -9.32465008e-01], [4.06271283e-01, 4.78221112e-02, -9.12500241e-01]],
        [[4.06271283e-01, 4.78221112e-02, -9.12500241e-01], [4.06736643e-01, 2.01762691e-16, -9.13545458e-01]],
        [[4.06736643e-01, 2.01762691e-16, -9.13545458e-01], [3.58367950e-01, 0.00000000e+00, -9.33580426e-01]]
    ])

    face_edges_lonlat = np.array(
        [[_xyz_to_lonlat_rad(*edge[0]), _xyz_to_lonlat_rad(*edge[1])] for edge in face_edges_cart])

    bounds = _populate_face_latlon_bound(face_edges_cart, face_edges_lonlat)
    expected_bounds = np.array([[-1.20427718, -1.14935491], [0, 0.13568803]])
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_near_pole2_latlon_bounds_gca():
    face_edges_cart = np.array([
        [[3.57939780e-01, -4.88684203e-02, -9.32465008e-01], [3.58367950e-01, 0.00000000e+00, -9.33580426e-01]],
        [[3.58367950e-01, 0.00000000e+00, -9.33580426e-01], [4.06736643e-01, 2.01762691e-16, -9.13545458e-01]],
        [[4.06736643e-01, 2.01762691e-16, -9.13545458e-01], [4.06271283e-01, -4.78221112e-02, -9.12500241e-01]],
        [[4.06271283e-01, -4.78221112e-02, -9.12500241e-01], [3.57939780e-01, -4.88684203e-02, -9.32465008e-01]]
    ])

    face_edges_lonlat = np.array(
        [[_xyz_to_lonlat_rad(*edge[0]), _xyz_to_lonlat_rad(*edge[1])] for edge in face_edges_cart])

    bounds = _populate_face_latlon_bound(face_edges_cart, face_edges_lonlat)
    expected_bounds = np.array([[-1.20427718, -1.14935491], [6.147497, 4.960524e-16]])
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_long_face_latlon_bounds_gca():
    face_edges_cart = np.array([
        [[9.9999946355819702e-01, -6.7040475551038980e-04, 8.0396590055897832e-04],
         [9.9999439716339111e-01, -3.2541253603994846e-03, -8.0110825365409255e-04]],
        [[9.9999439716339111e-01, -3.2541253603994846e-03, -8.0110825365409255e-04],
         [9.9998968839645386e-01, -3.1763643492013216e-03, -3.2474612817168236e-03]],
        [[9.9998968839645386e-01, -3.1763643492013216e-03, -3.2474612817168236e-03],
         [9.9998861551284790e-01, -8.2993711112067103e-04, -4.7004125081002712e-03]],
        [[9.9998861551284790e-01, -8.2993711112067103e-04, -4.7004125081002712e-03],
         [9.9999368190765381e-01, 1.7522916896268725e-03, -3.0944822356104851e-03]],
        [[9.9999368190765381e-01, 1.7522916896268725e-03, -3.0944822356104851e-03],
         [9.9999833106994629e-01, 1.6786820488050580e-03, -6.4892979571595788e-04]],
        [[9.9999833106994629e-01, 1.6786820488050580e-03, -6.4892979571595788e-04],
         [9.9999946355819702e-01, -6.7040475551038980e-04, 8.0396590055897832e-04]]
    ])

    face_edges_lonlat = np.array(
        [[_xyz_to_lonlat_rad(*edge[0]), _xyz_to_lonlat_rad(*edge[1])] for edge in face_edges_cart])

    bounds = _populate_face_latlon_bound(face_edges_cart, face_edges_lonlat)

    # The expected bounds should not contain the south pole [0,-0.5*np.pi]
    assert bounds[1][0] != 0.0


def test_populate_bounds_node_on_pole_latlon_bounds_gca():
    vertices_lonlat = [[10.0, 90.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
    lat_max = np.pi / 2
    lat_min = min(np.deg2rad(10.0),
                  np.asin(extreme_gca_z(np.array([vertices_cart[1], vertices_cart[2]]), extreme_type="min")))
    lon_min = np.deg2rad(10.0)
    lon_max = np.deg2rad(50.0)
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_gca(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_gca(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_edge_over_pole_latlon_bounds_gca():
    vertices_lonlat = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
    lat_max = np.pi / 2
    lat_min = min(np.deg2rad(60.0),
                  np.asin(extreme_gca_z(np.array([vertices_cart[1], vertices_cart[2]]), extreme_type="min")))
    lon_min = np.deg2rad(210.0)
    lon_max = np.deg2rad(30.0)
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_gca(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_gca(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_pole_inside_latlon_bounds_gca():
    vertices_lonlat = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
    lat_max = np.pi / 2
    lat_min = min(np.deg2rad(60.0),
                  np.asin(extreme_gca_z(np.array([vertices_cart[1], vertices_cart[2]]), extreme_type="min")))
    lon_min = 0
    lon_max = 2 * np.pi
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_gca(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_gca(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_latlonface(face_nodes_ind, face_edges_ind,
                                                                            edge_nodes_grid, node_x, node_y, node_z):
    """Construct an array to hold the edge Cartesian coordinates connectivity for a face in a grid."""
    mask = face_edges_ind != INT_FILL_VALUE
    valid_edges = face_edges_ind[mask]
    face_edges = edge_nodes_grid[valid_edges]

    face_edges[0] = [face_nodes_ind[0], face_nodes_ind[1]]

    for idx in range(1, len(face_edges)):
        if face_edges[idx][0] != face_edges[idx - 1][1]:
            face_edges[idx] = face_edges[idx][::-1]

    cartesian_coordinates = np.array(
        [
            [[node_x[node], node_y[node], node_z[node]] for node in edge]
            for edge in face_edges
        ]
    )

    return cartesian_coordinates


def _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_latlonface(face_nodes_ind, face_edges_ind,
                                                                             edge_nodes_grid, node_lon, node_lat):
    """Construct an array to hold the edge lat lon in radian connectivity for a face in a grid."""
    mask = face_edges_ind != INT_FILL_VALUE
    valid_edges = face_edges_ind[mask]
    face_edges = edge_nodes_grid[valid_edges]

    face_edges[0] = [face_nodes_ind[0], face_nodes_ind[1]]

    for idx in range(1, len(face_edges)):
        if face_edges[idx][0] != face_edges[idx - 1][1]:
            face_edges[idx] = face_edges[idx][::-1]

    lonlat_coordinates = np.array(
        [
            [
                [
                    np.mod(np.deg2rad(node_lon[node]), 2 * np.pi),
                    np.deg2rad(node_lat[node]),
                ]
                for node in edge
            ]
            for edge in face_edges
        ]
    )

    return lonlat_coordinates


def test_populate_bounds_normal_latlon_bounds_latlonface():
    vertices_lonlat = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    lat_max = np.deg2rad(60.0)
    lat_min = np.deg2rad(10.0)
    lon_min = np.deg2rad(10.0)
    lon_max = np.deg2rad(50.0)
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_latlonface(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_latlonface(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                         is_latlonface=True)
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_antimeridian_latlon_bounds_latlonface():
    vertices_lonlat = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    lat_max = np.deg2rad(60.0)
    lat_min = np.deg2rad(10.0)
    lon_min = np.deg2rad(350.0)
    lon_max = np.deg2rad(50.0)
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_latlonface(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_latlonface(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                         is_latlonface=True)
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_node_on_pole_latlon_bounds_latlonface():
    vertices_lonlat = [[10.0, 90.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    lat_max = np.pi / 2
    lat_min = np.deg2rad(10.0)
    lon_min = np.deg2rad(10.0)
    lon_max = np.deg2rad(50.0)
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_latlonface(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_latlonface(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                         is_latlonface=True)
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_edge_over_pole_latlon_bounds_latlonface():
    vertices_lonlat = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    lat_max = np.pi / 2
    lat_min = np.deg2rad(60.0)
    lon_min = np.deg2rad(210.0)
    lon_max = np.deg2rad(30.0)
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_latlonface(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_latlonface(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                         is_latlonface=True)
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_pole_inside_latlon_bounds_latlonface():
    vertices_lonlat = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    lat_max = np.pi / 2
    lat_min = np.deg2rad(60.0)
    lon_min = 0
    lon_max = 2 * np.pi
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_latlonface(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_latlonface(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                         is_latlonface=True)
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_gca_list(face_nodes_ind, face_edges_ind,
                                                                          edge_nodes_grid, node_x, node_y, node_z):
    """Construct an array to hold the edge Cartesian coordinates connectivity for a face in a grid."""
    mask = face_edges_ind != INT_FILL_VALUE
    valid_edges = face_edges_ind[mask]
    face_edges = edge_nodes_grid[valid_edges]

    face_edges[0] = [face_nodes_ind[0], face_nodes_ind[1]]

    for idx in range(1, len(face_edges)):
        if face_edges[idx][0] != face_edges[idx - 1][1]:
            face_edges[idx] = face_edges[idx][::-1]

    cartesian_coordinates = np.array(
        [
            [[node_x[node], node_y[node], node_z[node]] for node in edge]
            for edge in face_edges
        ]
    )

    return cartesian_coordinates


def _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_gca_list(face_nodes_ind, face_edges_ind,
                                                                           edge_nodes_grid, node_lon, node_lat):
    """Construct an array to hold the edge lat lon in radian connectivity for a face in a grid."""
    mask = face_edges_ind != INT_FILL_VALUE
    valid_edges = face_edges_ind[mask]
    face_edges = edge_nodes_grid[valid_edges]

    face_edges[0] = [face_nodes_ind[0], face_nodes_ind[1]]

    for idx in range(1, len(face_edges)):
        if face_edges[idx][0] != face_edges[idx - 1][1]:
            face_edges[idx] = face_edges[idx][::-1]

    lonlat_coordinates = np.array(
        [
            [
                [
                    np.mod(np.deg2rad(node_lon[node]), 2 * np.pi),
                    np.deg2rad(node_lat[node]),
                ]
                for node in edge
            ]
            for edge in face_edges
        ]
    )

    return lonlat_coordinates


def test_populate_bounds_normal_latlon_bounds_gca_list():
    vertices_lonlat = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    lat_max = np.deg2rad(60.0)
    lat_min = np.deg2rad(10.0)
    lon_min = np.deg2rad(10.0)
    lon_max = np.deg2rad(50.0)
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_gca_list(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_gca_list(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                         is_GCA_list=[True, False, True, False])
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_antimeridian_latlon_bounds_gca_list():
    vertices_lonlat = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    lat_max = np.deg2rad(60.0)
    lat_min = np.deg2rad(10.0)
    lon_min = np.deg2rad(350.0)
    lon_max = np.deg2rad(50.0)
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_gca_list(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_gca_list(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                         is_GCA_list=[True, False, True, False])
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_node_on_pole_latlon_bounds_gca_list():
    vertices_lonlat = [[10.0, 90.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    lat_max = np.pi / 2
    lat_min = np.deg2rad(10.0)
    lon_min = np.deg2rad(10.0)
    lon_max = np.deg2rad(50.0)
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_gca_list(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_gca_list(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                         is_GCA_list=[True, False, True, False])
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_edge_over_pole_latlon_bounds_gca_list():
    vertices_lonlat = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    lat_max = np.pi / 2
    lat_min = np.deg2rad(60.0)
    lon_min = np.deg2rad(210.0)
    lon_max = np.deg2rad(30.0)
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_gca_list(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_gca_list(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                         is_GCA_list=[True, False, True, False])
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_pole_inside_latlon_bounds_gca_list():
    vertices_lonlat = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    lat_max = np.pi / 2
    lat_min = np.deg2rad(60.0)
    lon_min = 0
    lon_max = 2 * np.pi
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    face_edges_connectivity_cartesian = _get_cartesian_face_edge_nodes_testcase_helper_latlon_bounds_gca_list(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_x.values,
        grid.node_y.values, grid.node_z.values)
    face_edges_connectivity_lonlat = _get_lonlat_rad_face_edge_nodes_testcase_helper_latlon_bounds_gca_list(
        grid.face_node_connectivity.values[0],
        grid.face_edge_connectivity.values[0],
        grid.edge_node_connectivity.values, grid.node_lon.values,
        grid.node_lat.values)
    expected_bounds = np.array([[lat_min, lat_max], [lon_min, lon_max]])
    bounds = _populate_face_latlon_bound(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat,
                                         is_GCA_list=[True, False, True, False])
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_GCA_mix_latlon_bounds_mix():
    face_1 = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    face_2 = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
    face_3 = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
    face_4 = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]

    faces = [face_1, face_2, face_3, face_4]

    expected_bounds = [[[0.17453293, 1.07370494], [0.17453293, 0.87266463]],
                       [[0.17453293, 1.10714872], [6.10865238, 0.87266463]],
                       [[1.04719755, 1.57079633], [3.66519143, 0.52359878]],
                       [[1.04719755, 1.57079633], [0., 6.28318531]]]

    grid = ux.Grid.from_face_vertices(faces, latlon=True)
    face_bounds = grid.bounds.values
    for i in range(len(faces)):
        np.testing.assert_allclose(face_bounds[i], expected_bounds[i], atol=ERROR_TOLERANCE)


def test_populate_bounds_LatlonFace_mix_latlon_bounds_mix():
    face_1 = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    face_2 = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
    face_3 = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
    face_4 = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]

    faces = [face_1, face_2, face_3, face_4]

    expected_bounds = [[[np.deg2rad(10.0), np.deg2rad(60.0)], [np.deg2rad(10.0), np.deg2rad(50.0)]],
                       [[np.deg2rad(10.0), np.deg2rad(60.0)], [np.deg2rad(350.0), np.deg2rad(50.0)]],
                       [[np.deg2rad(60.0), np.pi / 2], [np.deg2rad(210.0), np.deg2rad(30.0)]],
                       [[np.deg2rad(60.0), np.pi / 2], [0., 2 * np.pi]]]

    grid = ux.Grid.from_face_vertices(faces, latlon=True)
    bounds_xarray = _populate_bounds(grid, is_latlonface=True, return_array=True)
    face_bounds = bounds_xarray.values
    for i in range(len(faces)):
        np.testing.assert_allclose(face_bounds[i], expected_bounds[i], atol=ERROR_TOLERANCE)


def test_populate_bounds_GCAList_mix_latlon_bounds_mix():
    face_1 = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    face_2 = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
    face_3 = [[210.0, 80.0], [350.0, 60.0], [10.0, 60.0], [30.0, 80.0]]
    face_4 = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]

    faces = [face_1, face_2, face_3, face_4]

    expected_bounds = [[[np.deg2rad(10.0), np.deg2rad(60.0)], [np.deg2rad(10.0), np.deg2rad(50.0)]],
                       [[np.deg2rad(10.0), np.deg2rad(60.0)], [np.deg2rad(350.0), np.deg2rad(50.0)]],
                       [[np.deg2rad(60.0), np.pi / 2], [np.deg2rad(210.0), np.deg2rad(30.0)]],
                       [[np.deg2rad(60.0), np.pi / 2], [0., 2 * np.pi]]]

    grid = ux.Grid.from_face_vertices(faces, latlon=True)
    bounds_xarray = _populate_bounds(grid, is_face_GCA_list=np.array([[True, False, True, False],
                                                                      [True, False, True, False],
                                                                      [True, False, True, False],
                                                                      [True, False, True, False]]), return_array=True)
    face_bounds = bounds_xarray.values
    for i in range(len(faces)):
        np.testing.assert_allclose(face_bounds[i], expected_bounds[i], atol=ERROR_TOLERANCE)


def test_face_bounds_latlon_bounds_files():
    """Test to ensure ``Grid.face_bounds`` works correctly for all grid files."""
    for grid_path in grid_files_latlonBound:
        try:
            # Open the grid file
            uxgrid = ux.open_grid(grid_path)

            # Test: Ensure the bounds are obtained
            bounds = uxgrid.bounds
            assert bounds is not None, f"Grid.face_bounds should not be None for {grid_path}"

        except Exception as e:
            # Print the failing grid file and re-raise the exception
            print(f"Test failed for grid file: {grid_path}")
            raise e

        finally:
            # Clean up the grid object
            del uxgrid


def test_engine_geodataframe():
    uxgrid = ux.open_grid(gridfile_geoflow)
    for engine in ['geopandas', 'spatialpandas']:
        gdf = uxgrid.to_geodataframe(engine=engine)


def test_periodic_elements_geodataframe():
    uxgrid = ux.open_grid(gridfile_geoflow)
    for periodic_elements in ['ignore', 'exclude', 'split']:
        gdf = uxgrid.to_geodataframe(periodic_elements=periodic_elements)


def test_to_gdf_geodataframe():
    uxgrid = ux.open_grid(gridfile_geoflow)

    gdf_with_am = uxgrid.to_geodataframe(exclude_antimeridian=False)

    gdf_without_am = uxgrid.to_geodataframe(exclude_antimeridian=True)


def test_cache_and_override_geodataframe():
    """Tests the cache and override functionality for GeoDataFrame conversion."""
    uxgrid = ux.open_grid(gridfile_geoflow)

    gdf_a = uxgrid.to_geodataframe(exclude_antimeridian=False)

    gdf_b = uxgrid.to_geodataframe(exclude_antimeridian=False)

    assert gdf_a is gdf_b

    gdf_c = uxgrid.to_geodataframe(exclude_antimeridian=True)

    assert gdf_a is not gdf_c

    gdf_d = uxgrid.to_geodataframe(exclude_antimeridian=True)

    assert gdf_d is gdf_c

    gdf_e = uxgrid.to_geodataframe(exclude_antimeridian=True, override=True, cache=False)

    assert gdf_d is not gdf_e

    gdf_f = uxgrid.to_geodataframe(exclude_antimeridian=True)

    assert gdf_f is not gdf_e


# Test point_in_face function
def test_point_inside():
    """Test the function `point_in_face`, where the points are all inside the face"""

    # Open grid
    grid = ux.open_grid(grid_mpas_2)

    # Get the face edges of all faces in the grid
    faces_edges_cartesian = _get_cartesian_faces_edge_nodes(grid.face_node_connectivity.values, grid.n_face,
                                                            grid.n_max_face_edges, grid.node_x.values,
                                                            grid.node_y.values, grid.node_z.values)

    # Loop through each face
    for i in range(grid.n_face):
        # Set the point as the face center of the polygon
        point_xyz = np.array([grid.face_x[i].values, grid.face_y[i].values, grid.face_z[i].values])

        # Assert that the point is in the polygon
        assert point_in_face(faces_edges_cartesian[i], point_xyz, inclusive=True)


def test_point_outside():
    """Test the function `point_in_face`, where the point is outside the face"""

    # Open grid
    grid = ux.open_grid(grid_mpas_2)

    # Get the face edges of all faces in the grid
    faces_edges_cartesian = _get_cartesian_faces_edge_nodes(grid.face_node_connectivity.values, grid.n_face,
                                                            grid.n_max_face_edges, grid.node_x.values,
                                                            grid.node_y.values, grid.node_z.values)

    # Set the point as the face center of a different face than the face tested
    point_xyz = np.array([grid.face_x[1].values, grid.face_y[1].values, grid.face_z[1].values])

    # Assert that the point is not in the face tested
    assert not point_in_face(faces_edges_cartesian[0], point_xyz, inclusive=True)


def test_point_on_node():
    """Test the function `point_in_face`, when the point is on the node of the polygon"""

    # Open grid
    grid = ux.open_grid(grid_mpas_2)

    # Get the face edges of all faces in the grid
    faces_edges_cartesian = _get_cartesian_faces_edge_nodes(grid.face_node_connectivity.values, grid.n_face,
                                                            grid.n_max_face_edges, grid.node_x.values,
                                                            grid.node_y.values, grid.node_z.values)

    # Set the point as a node
    point_xyz = np.array([*faces_edges_cartesian[0][0][0]])

    # Assert that the point is in the face when inclusive is true
    assert point_in_face(faces_edges_cartesian[0], point_xyz, inclusive=True)

    # Assert that the point is not in the face when inclusive is false
    assert not point_in_face(faces_edges_cartesian[0], point_xyz, inclusive=False)


def test_point_inside_close():
    """Test the function `point_in_face`, where the point is inside the face, but very close to the edge"""

    # Create a square
    vertices_lonlat = [[-10.0, 10.0], [-10.0, -10.0], [10.0, -10.0], [10.0, 10.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    # Choose a point just inside the square
    point = np.array(_lonlat_rad_to_xyz(np.deg2rad(0.0), np.deg2rad(-9.8)))

    # Create the grid and face edges
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    faces_edges_cartesian = _get_cartesian_faces_edge_nodes(grid.face_node_connectivity.values, grid.n_face,
                                                            grid.n_max_face_edges, grid.node_x.values,
                                                            grid.node_y.values, grid.node_z.values)

    # Use point in face to determine if the point is inside or out of the face
    assert point_in_face(faces_edges_cartesian[0], point_xyz=point, inclusive=False)


def test_point_outside_close():
    """Test the function `point_in_face`, where the point is outside the face, but very close to the edge"""

    # Create a square
    vertices_lonlat = [[-10.0, 10.0], [-10.0, -10.0], [10.0, -10.0], [10.0, 10.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    # Choose a point just outside the square
    point = np.array(_lonlat_rad_to_xyz(np.deg2rad(0.0), np.deg2rad(-10.2)))

    # Create the grid and face edges
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    faces_edges_cartesian = _get_cartesian_faces_edge_nodes(grid.face_node_connectivity.values, grid.n_face,
                                                            grid.n_max_face_edges, grid.node_x.values,
                                                            grid.node_y.values, grid.node_z.values)

    # Use point in face to determine if the point is inside or out of the face
    assert not point_in_face(faces_edges_cartesian[0], point_xyz=point, inclusive=False)


def test_face_at_pole():
    """Test the function `point_in_face`, when the face is at the North Pole"""

    # Generate a face that is at a pole
    vertices_lonlat = [[10.0, 90.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    point = np.array(_lonlat_rad_to_xyz(np.deg2rad(25), np.deg2rad(30)))

    # Create the grid and face edges
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    faces_edges_cartesian = _get_cartesian_faces_edge_nodes(grid.face_node_connectivity.values, grid.n_face,
                                                            grid.n_max_face_edges, grid.node_x.values,
                                                            grid.node_y.values, grid.node_z.values)

    assert point_in_face(faces_edges_cartesian[0], point_xyz=point, inclusive=True)


def test_face_at_antimeridian():
    """Test the function `point_in_face`, where the face crosses the antimeridian"""

    # Generate a face crossing the antimeridian
    vertices_lonlat = [[350, 60.0], [350, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)
    point = np.array(_lonlat_rad_to_xyz(np.deg2rad(25), np.deg2rad(30)))

    # Create the grid and face edges
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    faces_edges_cartesian = _get_cartesian_faces_edge_nodes(grid.face_node_connectivity.values, grid.n_face,
                                                            grid.n_max_face_edges, grid.node_x.values,
                                                            grid.node_y.values, grid.node_z.values)

    assert point_in_face(faces_edges_cartesian[0], point_xyz=point, inclusive=True)


def test_face_normal_face():
    """Test the function `point_in_face`, where the face is a normal face, not crossing the antimeridian or the
    poles"""

    # Generate a normal face that is not crossing the antimeridian or the poles
    vertices_lonlat = [[10.0, 60.0], [10.0, 10.0], [50.0, 10.0], [50.0, 60.0]]
    vertices_lonlat = np.array(vertices_lonlat)
    point = np.array(_lonlat_rad_to_xyz(np.deg2rad(25), np.deg2rad(30)))

    # Create the grid and face edges
    grid = ux.Grid.from_face_vertices(vertices_lonlat, latlon=True)
    faces_edges_cartesian = _get_cartesian_faces_edge_nodes(grid.face_node_connectivity.values, grid.n_face,
                                                            grid.n_max_face_edges, grid.node_x.values,
                                                            grid.node_y.values, grid.node_z.values)

    assert point_in_face(faces_edges_cartesian[0], point_xyz=point, inclusive=True)


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


def test_haversine_distance_creation():
    """Tests the use of `haversine_distance`"""

    # Create two points
    point_a = [np.deg2rad(-34.8), np.deg2rad(-58.5)]
    point_b = [np.deg2rad(49.0), np.deg2rad(2.6)]

    result = haversine_distances([point_a, point_b])

    distance = haversine_distance(point_a[1], point_a[0], point_b[1], point_b[0])

    assert np.isclose(result[0][1], distance, atol=1e-6)
