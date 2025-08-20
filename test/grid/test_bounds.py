import os
import numpy as np
import numpy.testing as nt
import pytest

from pathlib import Path

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.arcs import extreme_gca_z
from uxarray.grid.bounds import _construct_face_bounds, insert_pt_in_latlonbox, _get_latlonbox_width

current_path = Path(os.path.dirname(os.path.realpath(__file__))).parent

grid_quad_hex = current_path / "meshfiles" / "ugrid" / "quad-hexagon" / "grid.nc"
grid_geoflow = current_path / "meshfiles" / "ugrid" / "geoflow-small" / "grid.nc"
gridfile_CSne8 = current_path / "meshfiles" / "scrip" / "outCSne8" / "outCSne8.nc"
grid_mpas = current_path / "meshfiles" / "mpas" / "QU" / "oQU480.231010.nc"

# List of grid files to test
grid_files_latlonBound = [grid_quad_hex, grid_geoflow, gridfile_CSne8, grid_mpas]


def test_get_latlonbox_width():
    gca_latlon = np.array([[0.0, 0.0], [0.0, 3.0]])
    width = _get_latlonbox_width(gca_latlon)
    assert width == 3.0

    gca_latlon = np.array([[0.0, 0.0], [2 * np.pi - 1.0, 1.0]])
    width = _get_latlonbox_width(gca_latlon)
    assert width == 2.0


def test_insert_pt_in_latlonbox_non_periodic():
    old_box = np.array([[0.1, 0.2], [0.3, 0.4]])  # Radians
    new_pt = np.array([0.15, 0.35])
    expected = np.array([[0.1, 0.2], [0.3, 0.4]])
    result = insert_pt_in_latlonbox(old_box, new_pt, False)
    np.testing.assert_array_equal(result, expected)


def test_insert_pt_in_latlonbox_periodic():
    old_box = np.array([[0.1, 0.2], [6.0, 0.1]])  # Radians, periodic
    new_pt = np.array([0.15, 6.2])
    expected = np.array([[0.1, 0.2], [6.0, 0.1]])
    result = insert_pt_in_latlonbox(old_box, new_pt, True)
    np.testing.assert_array_equal(result, expected)


def test_insert_pt_in_latlonbox_pole():
    old_box = np.array([[0.1, 0.2], [0.3, 0.4]])
    new_pt = np.array([np.pi / 2, np.nan])  # Pole point
    expected = np.array([[0.1, np.pi / 2], [0.3, 0.4]])
    result = insert_pt_in_latlonbox(old_box, new_pt)
    np.testing.assert_array_equal(result, expected)


def test_insert_pt_in_empty_state():
    old_box = np.array([[np.nan, np.nan],
                        [np.nan, np.nan]])  # Empty state
    new_pt = np.array([0.15, 0.35])
    expected = np.array([[0.15, 0.15], [0.35, 0.35]])
    result = insert_pt_in_latlonbox(old_box, new_pt)
    np.testing.assert_array_equal(result, expected)


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
                [node_lon[node], node_lat[node]] for node in edge
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
