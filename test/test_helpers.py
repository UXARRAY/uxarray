import os
import numpy as np
import numpy.testing as nt
import random
import xarray as xr
from pathlib import Path
import pytest

import uxarray as ux
from uxarray.grid.connectivity import _replace_fill_values
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _normalize_xyz, _xyz_to_lonlat_rad
from uxarray.grid.arcs import point_within_gca, in_between
from uxarray.grid.utils import _get_cartesian_faces_edge_nodes, _get_lonlat_rad_faces_edge_nodes, _angle_of_2_vectors
from uxarray.grid.geometry import pole_point_inside_polygon

try:
    import constants
except ImportError:
    from . import constants

# Data files
current_path = Path(os.path.dirname(os.path.realpath(__file__)))

gridfile_exo_CSne8 = current_path / "meshfiles" / "exodus" / "outCSne8" / "outCSne8.g"
gridfile_scrip_CSne8 = current_path / 'meshfiles' / "scrip" / "outCSne8" / 'outCSne8.nc'
gridfile_geoflowsmall_grid = current_path / 'meshfiles' / "ugrid" / "geoflow-small" / 'grid.nc'
gridfile_geoflowsmall_var = current_path / 'meshfiles' / "ugrid" / "geoflow-small" / 'v1.nc'

err_tolerance = 1.0e-12

def test_face_area_coords():
    """Test function for helper function get_all_face_area_from_coords."""
    # Note: currently only testing one face, but this can be used to get area of multiple faces
    # Cartesian coordinates (x, y, z) for each city
    # Index 0: Chicago, Index 1: Miami, Index 2: Newburgh, New York, USA.
    x = np.array([0.02974582, 0.1534193, 0.18363692])
    y = np.array([-0.74469018, -0.88744577, -0.72230586])
    z = np.array([0.66674712, 0.43462917, 0.66674712])
    face_nodes = np.array([[0, 1, 2]])
    face_dimension = np.array([3], dtype=INT_DTYPE)

    area, _ = ux.grid.area.get_all_face_area_from_coords(
        x, y, z, face_nodes, face_dimension, 3, coords_type="cartesian")
    nt.assert_almost_equal(area, constants.TRI_AREA, decimal=5)


def test_calculate_face_area():
    """Test function for helper function calculate_face_area - only one face."""
    # Note: currently only testing one face, but this can be used to get area of multiple faces
    # Also note, this does not need face_nodes, assumes nodes are in counterclockwise orientation
    x = np.array([0.02974582, 0.1534193, 0.18363692])
    y = np.array([-0.74469018, -0.88744577, -0.72230586])
    z = np.array([0.66674712, 0.43462917, 0.66674712])

    area, _ = ux.grid.area.calculate_face_area(
        x, y, z, "gaussian", 5, "cartesian", latitude_adjusted_area=False)

    nt.assert_almost_equal(area, constants.TRI_AREA, decimal=5)

    area_corrected, _ = ux.grid.area.calculate_face_area(
        x, y, z, "gaussian", 5, "cartesian", latitude_adjusted_area=True)

    nt.assert_almost_equal(area_corrected, constants.CORRECTED_TRI_AREA, decimal=5)

    # Make the same grid using lon/lat check area = constants.TRI_AREA
    lon = np.array([-87.7126, -80.1918, -75.7355])
    lat = np.array([41.8165, 25.7617, 41.8165])
    face_nodes = np.array([[0, 1, 2]])

    grid = ux.Grid.from_topology(
        node_lon=lon,
        node_lat=lat,
        face_node_connectivity=face_nodes,
        fill_value=-1,
    )

    area, _ = grid.compute_face_areas()
    nt.assert_almost_equal(area, constants.TRI_AREA, decimal=5)

def test_quadrature():
    order = 1
    dG, dW = ux.grid.area.get_tri_quadrature_dg(order)
    G = np.array([[0.33333333, 0.33333333, 0.33333333]])
    W = np.array([1.0])

    np.testing.assert_array_almost_equal(G, dG)
    np.testing.assert_array_almost_equal(W, dW)

    dG, dW = ux.grid.area.get_gauss_quadrature_dg(order)

    G = np.array([[0.5]])
    W = np.array([1.0])

    np.testing.assert_array_almost_equal(G, dG)
    np.testing.assert_array_almost_equal(W, dW)

def test_grid_center():
    """Calculates if the calculated center point of a grid box is the same
    as a given value for the same dataset."""
    ds_scrip_CSne8 = xr.open_dataset(gridfile_scrip_CSne8)

    # select actual center_lat/lon
    scrip_center_lon = ds_scrip_CSne8['grid_center_lon']
    scrip_center_lat = ds_scrip_CSne8['grid_center_lat']

    # Calculate the center_lat/lon using same dataset's corner_lat/lon
    calc_center = ux.io._scrip.grid_center_lat_lon(ds_scrip_CSne8)
    calc_lat = calc_center[0]
    calc_lon = calc_center[1]

    # Test that calculated center_lat/lon is the same as actual center_lat/lon
    np.testing.assert_array_almost_equal(scrip_center_lat, calc_lat)
    np.testing.assert_array_almost_equal(scrip_center_lon, calc_lon)

def test_normalize_in_place():
    x, y, z = _normalize_xyz(
        random.random(), random.random(),
        random.random())

    assert np.absolute(np.sqrt(x * x + y * y + z * z) - 1) <= err_tolerance

def test_node_xyz_to_lonlat_rad():
    x, y, z = _normalize_xyz(*[
        random.uniform(-1, 1),
        random.uniform(-1, 1),
        random.uniform(-1, 1)
    ])

    lon, lat = _xyz_to_lonlat_rad(x, y, z)
    new_x, new_y, new_z = _lonlat_rad_to_xyz(lon, lat)

    assert np.absolute(new_x - x) <= err_tolerance
    assert np.absolute(new_y - y) <= err_tolerance
    assert np.absolute(new_z - z) <= err_tolerance

def test_node_latlon_rad_to_xyz():
    lon, lat = [
        random.uniform(0, 2 * np.pi),
        random.uniform(-0.5 * np.pi, 0.5 * np.pi)
    ]

    x, y, z = _lonlat_rad_to_xyz(lon, lat)
    new_lon, new_lat = _xyz_to_lonlat_rad(x, y, z)

    assert np.absolute(new_lon - lon) <= err_tolerance
    assert np.absolute(new_lat - lat) <= err_tolerance

def test_invalid_indexing():
    """Tests if the current INT_DTYPE and INT_FILL_VALUE throw the correct
    errors when indexing."""
    dummy_data = np.array([1, 2, 3, 4])

    invalid_indices = np.array([INT_FILL_VALUE, INT_FILL_VALUE], dtype=INT_DTYPE)
    invalid_index = INT_FILL_VALUE

    # invalid index/indices should throw an Index Error
    with pytest.raises(IndexError):
        dummy_data[invalid_indices]
        dummy_data[invalid_index]

def test_replace_fill_values():
    """Tests _replace_fill_values() helper function across multiple
    different dtype arrays used as face_nodes."""

    # expected output from _replace_fill_values()
    face_nodes_gold = np.array(
        [[1, 2, INT_FILL_VALUE], [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE]],
        dtype=INT_DTYPE)

    # test different datatypes for face_nodes
    dtypes = [np.int32, np.int64, np.float32, np.float64]
    for dtype in dtypes:
        # test face nodes with set dtype
        face_nodes = np.array([[1, 2, -1], [-1, -1, -1]], dtype=dtype)
        face_nodes = xr.DataArray(data=face_nodes)

        # output of _replace_fill_values()
        face_nodes_test = _replace_fill_values(
            grid_var=face_nodes,
            original_fill=-1,
            new_fill=INT_FILL_VALUE,
            new_dtype=INT_DTYPE
        )

        assert np.array_equal(face_nodes_test, face_nodes_gold)

def test_replace_fill_values_invalid():
    """Tests _replace_fill_values() helper function attempting to use a
    fill value that is not representable by the current dtype."""

    face_nodes = np.array([[1, 2, -1], [-1, -1, -1]], dtype=np.int32)

    # invalid fill value with dtype should raise a valueError
    with pytest.raises(ValueError):
        # INT_FILL_VALUE (max(uint32) not representable by int16)
        face_nodes_test = _replace_fill_values(
            grid_var=face_nodes,
            original_fill=-1,
            new_fill=INT_FILL_VALUE,
            new_dtype=np.int16
        )

def test_convert_face_node_conn_to_sparse_matrix():
    """Tests _face_nodes_to_sparse_matrix() helper function to see if can
    generate sparse matrix from face_nodes_conn that has Fill Values."""
    face_nodes_conn = np.array([[3, 4, 5, INT_FILL_VALUE], [3, 0, 2, 5],
                                [3, 4, 1, 0], [0, 1, 2, INT_FILL_VALUE]])

    face_indices, nodes_indices, non_zero_flag = ux.grid.connectivity._face_nodes_to_sparse_matrix(
        face_nodes_conn)
    expected_non_zero_flag = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    expected_face_indices = np.array(
        [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    expected_nodes_indices = np.array(
        [3, 4, 5, 3, 0, 2, 5, 3, 4, 1, 0, 0, 1, 2])

    nt.assert_array_equal(non_zero_flag, expected_non_zero_flag)
    nt.assert_array_equal(face_indices, expected_face_indices)
    nt.assert_array_equal(nodes_indices, expected_nodes_indices)

def test_in_between():
    # Test the in_between operator
    assert in_between(0, 1, 2)
    assert in_between(-1, -1.5, -2)

def test_angle_of_2_vectors():
    # Test the angle between two vectors
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    assert pytest.approx(_angle_of_2_vectors(v1, v2)) == np.pi / 2.0

    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    assert pytest.approx(_angle_of_2_vectors(v1, v2)) == 0.0

def test_angle_of_2_vectors_180_degree():
    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(0.0),
                                np.deg2rad(0.0)),
        _lonlat_rad_to_xyz(np.deg2rad(181.0),
                                np.deg2rad(0.0))
    ])

    res = _angle_of_2_vectors(GCR1_cart[0], GCR1_cart[1])

    # The angle between the two vectors should be 181 degree
    assert pytest.approx(res, abs=1e-8) == np.deg2rad(181.0)

    GCR1_cart = np.array([
        _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                np.deg2rad(89.0)),
        _lonlat_rad_to_xyz(np.deg2rad(170.0),
                                np.deg2rad(-10.0))
    ])

    res = _angle_of_2_vectors(GCR1_cart[0], GCR1_cart[1])

    # The angle between the two vectors should be 99 degrees
    assert pytest.approx(res, abs=1e-8) == np.deg2rad(89.0+10.0)

def test_get_cartesian_face_edge_nodes_pipeline():
    vertices = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]]
    vertices = [x / np.linalg.norm(x) for x in vertices]
    grid = ux.Grid.from_face_vertices(vertices, latlon=False)

    face_node_conn = grid.face_node_connectivity.values
    n_nodes_per_face = np.array([len(face) for face in face_node_conn])
    n_face = len(face_node_conn)
    n_max_face_edges = max(n_nodes_per_face)
    node_x = grid.node_x.values
    node_y = grid.node_y.values
    node_z = grid.node_z.values

    faces_edges_connectivity_cartesian = _get_cartesian_faces_edge_nodes(face_node_conn, n_face, n_max_face_edges,
                                                                        node_x, node_y, node_z)

    face_edges_xyz = faces_edges_connectivity_cartesian[0]

    x = face_edges_xyz[:, :, 0]
    y = face_edges_xyz[:, :, 1]
    z = face_edges_xyz[:, :, 2]

    lon, lat = _xyz_to_lonlat_rad(x, y, z)

    face_edges_lonlat = np.stack((lon, lat), axis=2)
    result = pole_point_inside_polygon(
        1, face_edges_xyz, face_edges_lonlat
    )

    assert result is True

def test_get_cartesian_face_edge_nodes_filled_value():
    vertices = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]]
    vertices = [x / np.linalg.norm(x) for x in vertices]
    vertices.append([INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE])

    grid = ux.Grid.from_face_vertices(vertices, latlon=False)

    face_node_conn = grid.face_node_connectivity.values
    n_nodes_per_face = np.array([len(face) for face in face_node_conn])
    n_face = len(face_node_conn)
    n_max_face_edges = max(n_nodes_per_face)
    node_x = grid.node_x.values
    node_y = grid.node_y.values
    node_z = grid.node_z.values

    faces_edges_connectivity_cartesian = _get_cartesian_faces_edge_nodes(face_node_conn, n_face, n_max_face_edges,
                                                                        node_x, node_y, node_z)
    face_edges_xyz = faces_edges_connectivity_cartesian[0]

    x = face_edges_xyz[:, :, 0]
    y = face_edges_xyz[:, :, 1]
    z = face_edges_xyz[:, :, 2]

    lon, lat = _xyz_to_lonlat_rad(x, y, z)

    face_edges_lonlat = np.stack((lon, lat), axis=2)

    result = pole_point_inside_polygon(
        1, face_edges_xyz, face_edges_lonlat
    )

    assert result is True

def test_get_cartesian_face_edge_nodes_filled_value2():
    v0_deg = [10,10]
    v1_deg = [15,15]
    v2_deg = [5,15]
    v3_deg = [15,45]
    v4_deg = [5,45]

    v0_rad = np.deg2rad(v0_deg)
    v1_rad = np.deg2rad(v1_deg)
    v2_rad = np.deg2rad(v2_deg)
    v3_rad = np.deg2rad(v3_deg)
    v4_rad = np.deg2rad(v4_deg)

    v0_cart = _lonlat_rad_to_xyz(v0_rad[0],v0_rad[1])
    v1_cart = _lonlat_rad_to_xyz(v1_rad[0],v1_rad[1])
    v2_cart = _lonlat_rad_to_xyz(v2_rad[0],v2_rad[1])
    v3_cart = _lonlat_rad_to_xyz(v3_rad[0],v3_rad[1])
    v4_cart = _lonlat_rad_to_xyz(v4_rad[0],v4_rad[1])

    face_node_conn = np.array([[0, 1, 2, INT_FILL_VALUE],[1, 3, 4, 2]])
    n_face = 2
    n_max_face_edges = 4
    n_nodes_per_face = np.array([len(face) for face in face_node_conn])
    node_x = np.array([v0_cart[0],v1_cart[0],v2_cart[0],v3_cart[0],v4_cart[0]])
    node_y = np.array([v0_cart[1],v1_cart[1],v2_cart[1],v3_cart[1],v4_cart[1]])
    node_z = np.array([v0_cart[2],v1_cart[2],v2_cart[2],v3_cart[2],v4_cart[2]])

    face_edges_connectivity_cartesian = _get_cartesian_faces_edge_nodes(face_node_conn, n_face, n_max_face_edges,
                                                                        node_x, node_y, node_z)

    correct_result = np.array([
        [
            [[v0_cart[0], v0_cart[1], v0_cart[2]], [v1_cart[0], v1_cart[1], v1_cart[2]]],
            [[v1_cart[0], v1_cart[1], v1_cart[2]], [v2_cart[0], v2_cart[1], v2_cart[2]]],
            [[v2_cart[0], v2_cart[1], v2_cart[2]], [v0_cart[0], v0_cart[1], v0_cart[2]]],
            [[INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE], [INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE]]
        ],
        [
            [[v1_cart[0], v1_cart[1], v1_cart[2]], [v3_cart[0], v3_cart[1], v3_cart[2]]],
            [[v3_cart[0], v3_cart[1], v3_cart[2]], [v4_cart[0], v4_cart[1], v4_cart[2]]],
            [[v4_cart[0], v4_cart[1], v4_cart[2]], [v2_cart[0], v2_cart[1], v2_cart[2]]],
            [[v2_cart[0], v2_cart[1], v2_cart[2]], [v1_cart[0], v1_cart[1], v1_cart[2]]]
        ]
    ])

    assert face_edges_connectivity_cartesian.shape == correct_result.shape

def test_get_lonlat_face_edge_nodes_pipeline():
    vertices = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]]
    vertices = [x / np.linalg.norm(x) for x in vertices]
    grid = ux.Grid.from_face_vertices(vertices, latlon=False)

    face_node_conn = grid.face_node_connectivity.values
    n_nodes_per_face = np.array([len(face) for face in face_node_conn])
    n_face = len(face_node_conn)
    n_max_face_edges = max(n_nodes_per_face)
    node_lon = grid.node_lon.values
    node_lat = grid.node_lat.values

    face_edges_connectivity_lonlat = _get_lonlat_rad_faces_edge_nodes(face_node_conn, n_face, n_max_face_edges,
                                                                      node_lon, node_lat)

    face_edges_connectivity_lonlat = face_edges_connectivity_lonlat[0]
    face_edges_connectivity_cartesian = []
    for edge in face_edges_connectivity_lonlat:
        edge_cart = [_lonlat_rad_to_xyz(*node) for node in edge]
        face_edges_connectivity_cartesian.append(edge_cart)
    pass


    result = pole_point_inside_polygon(
        1, np.array(face_edges_connectivity_cartesian),face_edges_connectivity_lonlat
    )

    assert result is True

def test_get_lonlat_face_edge_nodes_filled_value():
    vertices = [[0.5, 0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5]]
    vertices = [x / np.linalg.norm(x) for x in vertices]
    vertices.append([INT_FILL_VALUE, INT_FILL_VALUE, INT_FILL_VALUE])

    grid = ux.Grid.from_face_vertices(vertices, latlon=False)

    face_node_conn = grid.face_node_connectivity.values
    n_nodes_per_face = np.array([len(face) for face in face_node_conn])
    n_face = len(face_node_conn)
    n_max_face_edges = max(n_nodes_per_face)
    node_lon = grid.node_lon.values
    node_lat = grid.node_lat.values

    face_edges_connectivity_lonlat = _get_lonlat_rad_faces_edge_nodes(face_node_conn, n_face, n_max_face_edges,
                                                                      node_lon, node_lat)

    face_edges_connectivity_lonlat = face_edges_connectivity_lonlat[0]
    face_edges_connectivity_cartesian = []
    for edge in face_edges_connectivity_lonlat:
        edge_cart = [_lonlat_rad_to_xyz(*node) for node in edge]
        face_edges_connectivity_cartesian.append(edge_cart)

    result = pole_point_inside_polygon(
        1, np.array(face_edges_connectivity_cartesian),face_edges_connectivity_lonlat
    )

    assert result is True

def test_get_lonlat_face_edge_nodes_filled_value2():
    v0_deg = [10,10]
    v1_deg = [15,15]
    v2_deg = [5,15]
    v3_deg = [15,45]
    v4_deg = [5,45]

    v0_rad = np.deg2rad(v0_deg)
    v1_rad = np.deg2rad(v1_deg)
    v2_rad = np.deg2rad(v2_deg)
    v3_rad = np.deg2rad(v3_deg)
    v4_rad = np.deg2rad(v4_deg)

    face_node_conn = np.array([[0, 1, 2, INT_FILL_VALUE],[1, 3, 4, 2]])
    n_face = 2
    n_max_face_edges = 4
    n_nodes_per_face = np.array([len(face) for face in face_node_conn])
    node_lon = np.array([v0_rad[0],v1_rad[0],v2_rad[0],v3_rad[0],v4_rad[0]])
    node_lat = np.array([v0_rad[1],v1_rad[1],v2_rad[1],v3_rad[1],v4_rad[1]])

    face_edges_connectivity_lonlat = _get_lonlat_rad_faces_edge_nodes(face_node_conn, n_face, n_max_face_edges,
                                                                      node_lon, node_lat)

    correct_result = np.array([
        [
            [[v0_rad[0], v0_rad[1]], [v1_rad[0], v1_rad[1]]],
            [[v1_rad[0], v1_rad[1]], [v2_rad[0], v2_rad[1]]],
            [[v2_rad[0], v2_rad[1]], [v0_rad[0], v0_rad[1]]],
            [[INT_FILL_VALUE, INT_FILL_VALUE], [INT_FILL_VALUE, INT_FILL_VALUE]]
        ],
        [
            [[v1_rad[0], v1_rad[1]], [v3_rad[0], v3_rad[1]]],
            [[v3_rad[0], v3_rad[1]], [v4_rad[0], v4_rad[1]]],
            [[v4_rad[0], v4_rad[1]], [v2_rad[0], v2_rad[1]]],
            [[v2_rad[0], v2_rad[1]], [v1_rad[0], v1_rad[1]]]
        ]
    ])

    assert face_edges_connectivity_lonlat.shape == correct_result.shape
