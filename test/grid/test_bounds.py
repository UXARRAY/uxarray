"""Tests for grid bounds calculation functionality.

This module contains tests for computing face bounds including:
- Normal bounds calculations
- Antimeridian handling in bounds
- Pole handling in bounds
- GCA (Great Circle Arc) bounds
- LatLonFace bounds
- Mixed bounds scenarios
"""

import numpy as np
import numpy.testing as nt

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _xyz_to_lonlat_rad
from uxarray.grid.arcs import extreme_gca_z
from uxarray.grid.bounds import _construct_face_bounds


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


def test_populate_bounds_equator_latlon_bounds_gca():
    face_edges_cart = np.array([
        [[0.99726469, -0.05226443, -0.05226443], [0.99862953, 0.0, -0.05233596]],
        [[0.99862953, 0.0, -0.05233596], [1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.99862953, -0.05233596, 0.0]],
        [[0.99862953, -0.05233596, 0.0], [0.99726469, -0.05226443, -0.05226443]]
    ])
    face_edges_lonlat = np.array(
        [[_xyz_to_lonlat_rad(*edge[0]), _xyz_to_lonlat_rad(*edge[1])] for edge in face_edges_cart])

    bounds = _construct_face_bounds(face_edges_cart, face_edges_lonlat)
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

    bounds = _construct_face_bounds(face_edges_cart, face_edges_lonlat)
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

    bounds = _construct_face_bounds(face_edges_cart, face_edges_lonlat)
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

    bounds = _construct_face_bounds(face_edges_cart, face_edges_lonlat)
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

    bounds = _construct_face_bounds(face_edges_cart, face_edges_lonlat)

    # The expected bounds should not contain the south pole [0,-0.5*np.pi]
    assert bounds[1][0] != 0.0


def test_face_bounds_latlon_bounds_files(gridpath):
    """Test face bounds calculation for various grid files."""
    grid_files = [
        gridpath("ugrid", "outCSne30", "outCSne30.ug"),
        gridpath("ugrid", "outRLL1deg", "outRLL1deg.ug"),
        gridpath("ugrid", "geoflow-small", "grid.nc"),
        gridpath("mpas", "QU", "mesh.QU.1920km.151026.nc"),
        gridpath("scrip", "outCSne8", "outCSne8.nc")
    ]

    for grid_file in grid_files:
        uxgrid = ux.open_grid(grid_file)
        bounds = uxgrid.bounds

        # Check that bounds array has correct shape
        assert bounds.shape == (uxgrid.n_face, 2, 2)

        # Check that latitude bounds are within valid range
        assert np.all(bounds[:, 0, 0] >= -np.pi/2)  # min lat >= -90°
        assert np.all(bounds[:, 0, 1] <= np.pi/2)   # max lat <= 90°

        # Check that longitude bounds are within valid range
        assert np.all(bounds[:, 1, 0] >= 0)         # min lon >= 0°
        assert np.all(bounds[:, 1, 1] <= 2*np.pi)   # max lon <= 360°

        # Check that min <= max for each bound
        assert np.all(bounds[:, 0, 0] <= bounds[:, 0, 1])  # lat_min <= lat_max
        # Note: longitude bounds can wrap around antimeridian, so we don't check lon_min <= lon_max
