import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _normalize_xyz, _xyz_to_lonlat_rad
from uxarray.grid.arcs import extreme_gca_z
from uxarray.grid.bounds import _construct_face_bounds, _populate_face_bounds


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
    bounds = _construct_face_bounds(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
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
    bounds = _construct_face_bounds(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
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

    bounds = _construct_face_bounds(face_edges_cart, face_edges_lonlat)
    expected_bounds = np.array([[-1.20427718, -1.14935491], [6.147497, 4.960524e-16]])
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


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
    bounds = _construct_face_bounds(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
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
    bounds = _construct_face_bounds(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)


def test_populate_bounds_pole_inside_latlon_bounds_gca():
    vertices_lonlat = [[200.0, 80.0], [350.0, 60.0], [10.0, 60.0], [40.0, 80.0]]
    vertices_lonlat = np.array(vertices_lonlat)

    vertices_rad = np.radians(vertices_lonlat)
    vertices_cart = np.vstack([_lonlat_rad_to_xyz(vertices_rad[:, 0], vertices_rad[:, 1])]).T
    lat_max = np.pi / 2
    lat_min = min(np.deg2rad(60.0),
                  np.asin(extreme_gca_z(np.array([vertices_cart[1], vertices_cart[2]]), extreme_type="min")))
    lon_min = 0.0
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
    bounds = _construct_face_bounds(face_edges_connectivity_cartesian, face_edges_connectivity_lonlat)
    np.testing.assert_allclose(bounds, expected_bounds, atol=ERROR_TOLERANCE)