import numpy as np
import numpy.testing as nt
import pandas as pd
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.utils import _get_cartesian_face_edge_nodes_array
from uxarray.grid.integrate import _zonal_face_weights, _zonal_face_weights_robust


def test_get_zonal_faces_weight_at_constLat_equator():
    face_0 = [[1.7 * np.pi, 0.25 * np.pi], [1.7 * np.pi, 0.0],
              [0.3 * np.pi, 0.0], [0.3 * np.pi, 0.25 * np.pi]]
    face_1 = [[0.3 * np.pi, 0.0], [0.3 * np.pi, -0.25 * np.pi],
              [0.6 * np.pi, -0.25 * np.pi], [0.6 * np.pi, 0.0]]
    face_2 = [[0.3 * np.pi, 0.25 * np.pi], [0.3 * np.pi, 0.0], [np.pi, 0.0],
              [np.pi, 0.25 * np.pi]]
    face_3 = [[0.7 * np.pi, 0.0], [0.7 * np.pi, -0.25 * np.pi],
              [np.pi, -0.25 * np.pi], [np.pi, 0.0]]

    # Convert the face vertices to xyz coordinates
    face_0 = [_lonlat_rad_to_xyz(*v) for v in face_0]
    face_1 = [_lonlat_rad_to_xyz(*v) for v in face_1]
    face_2 = [_lonlat_rad_to_xyz(*v) for v in face_2]
    face_3 = [_lonlat_rad_to_xyz(*v) for v in face_3]

    face_0_edge_nodes = np.array([[face_0[0], face_0[1]],
                                  [face_0[1], face_0[2]],
                                  [face_0[2], face_0[3]],
                                  [face_0[3], face_0[0]]])
    face_1_edge_nodes = np.array([[face_1[0], face_1[1]],
                                  [face_1[1], face_1[2]],
                                  [face_1[2], face_1[3]],
                                  [face_1[3], face_1[0]]])
    face_2_edge_nodes = np.array([[face_2[0], face_2[1]],
                                  [face_2[1], face_2[2]],
                                  [face_2[2], face_2[3]],
                                  [face_2[3], face_2[0]]])
    face_3_edge_nodes = np.array([[face_3[0], face_3[1]],
                                  [face_3[1], face_3[2]],
                                  [face_3[2], face_3[3]],
                                  [face_3[3], face_3[0]]])

    face_0_latlon_bound = np.array([[0.0, 0.25 * np.pi],
                                    [1.7 * np.pi, 0.3 * np.pi]])
    face_1_latlon_bound = np.array([[-0.25 * np.pi, 0.0],
                                    [0.3 * np.pi, 0.6 * np.pi]])
    face_2_latlon_bound = np.array([[0.0, 0.25 * np.pi],
                                    [0.3 * np.pi, np.pi]])
    face_3_latlon_bound = np.array([[-0.25 * np.pi, 0.0],
                                    [0.7 * np.pi, np.pi]])

    latlon_bounds = np.array([
        face_0_latlon_bound, face_1_latlon_bound, face_2_latlon_bound,
        face_3_latlon_bound
    ])

    face_edges_cart = np.array([
        face_0_edge_nodes, face_1_edge_nodes, face_2_edge_nodes,
        face_3_edge_nodes
    ])

    constLat_cart = 0.0

    weights = _zonal_face_weights(face_edges_cart,
                                  latlon_bounds,
                                  np.array([4, 4, 4, 4]),
                                  z=constLat_cart,
                                  check_equator=True)

    expected_weights = np.array([0.46153, 0.11538, 0.30769, 0.11538])

    nt.assert_array_almost_equal(weights, expected_weights, decimal=3)

    # A error will be raise if we don't set is_latlonface=True since the face_2 will be concave if
    # It's edges are all GCA
    with pytest.raises(ValueError):
        _zonal_face_weights_robust(np.array([
            face_0_edge_nodes, face_1_edge_nodes, face_2_edge_nodes
        ]), np.deg2rad(20), latlon_bounds)


def test_get_zonal_faces_weight_at_constLat_regular():
    face_0 = [[1.7 * np.pi, 0.25 * np.pi], [1.7 * np.pi, 0.0],
              [0.3 * np.pi, 0.0], [0.3 * np.pi, 0.25 * np.pi]]
    face_1 = [[0.4 * np.pi, 0.3 * np.pi], [0.4 * np.pi, 0.0],
              [0.5 * np.pi, 0.0], [0.5 * np.pi, 0.3 * np.pi]]

    # Convert the face vertices to xyz coordinates
    face_0 = [_lonlat_rad_to_xyz(*v) for v in face_0]
    face_1 = [_lonlat_rad_to_xyz(*v) for v in face_1]

    face_0_edge_nodes = np.array([[face_0[0], face_0[1]],
                                  [face_0[1], face_0[2]],
                                  [face_0[2], face_0[3]],
                                  [face_0[3], face_0[0]]])
    face_1_edge_nodes = np.array([[face_1[0], face_1[1]],
                                  [face_1[1], face_1[2]],
                                  [face_1[2], face_1[3]],
                                  [face_1[3], face_1[0]]])

    face_0_latlon_bound = np.array([[0.0, 0.25 * np.pi],
                                    [1.7 * np.pi, 0.3 * np.pi]])
    face_1_latlon_bound = np.array([[0.0, 0.3 * np.pi],
                                    [0.4 * np.pi, 0.5 * np.pi]])

    latlon_bounds = np.array([face_0_latlon_bound, face_1_latlon_bound])

    face_edges_cart = np.array([face_0_edge_nodes, face_1_edge_nodes])

    constLat_cart = np.sin(0.1 * np.pi)

    weights = _zonal_face_weights(face_edges_cart,
                                  latlon_bounds,
                                  np.array([4, 4]),
                                  z=constLat_cart)

    expected_weights = np.array([0.9, 0.1])

    nt.assert_array_almost_equal(weights, expected_weights, decimal=3)


def test_get_zonal_faces_weight_at_constLat_on_pole_one_face():
    # The face is touching the pole, so the weight should be 1.0 since there's only 1 face
    face_edges_cart = np.array([[
        [[-5.22644277e-02, -5.22644277e-02, -9.97264689e-01],
         [-5.23359562e-02, -6.40930613e-18, -9.98629535e-01]],
        [[-5.23359562e-02, -6.40930613e-18, -9.98629535e-01],
         [6.12323400e-17, 0.00000000e+00, -1.00000000e+00]],
        [[6.12323400e-17, 0.00000000e+00, -1.00000000e+00],
         [3.20465306e-18, -5.23359562e-02, -9.98629535e-01]],
        [[3.20465306e-18, -5.23359562e-02, -9.98629535e-01],
         [-5.22644277e-02, -5.22644277e-02, -9.97264689e-01]]
    ]])

    # Corrected face_bounds
    face_bounds = np.array([
        [-1.57079633, -1.4968158],
        [3.14159265, 0.]
    ])
    constLat_cart = -1

    weights = _zonal_face_weights(face_edges_cart,
                                  np.array([face_bounds]),
                                  np.array([4]),
                                  z=constLat_cart)

    expected_weights = np.array([1.0])

    nt.assert_array_equal(weights, expected_weights)


def test_get_zonal_faces_weight_at_constLat_on_pole_faces():
    # There will be 4 faces touching the pole, so the weight should be 0.25 for each face
    face_edges_cart = np.array([
        [
            [[5.22644277e-02, -5.22644277e-02, 9.97264689e-01], [5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
            [[5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
            [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [3.20465306e-18, -5.23359562e-02, 9.98629535e-01]],
            [[3.20465306e-18, -5.23359562e-02, 9.98629535e-01], [5.22644277e-02, -5.22644277e-02, 9.97264689e-01]]
        ],
        [
            [[3.20465306e-18, -5.23359562e-02, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
            [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [-5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
            [[-5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [-5.22644277e-02, -5.22644277e-02, 9.97264689e-01]],
            [[-5.22644277e-02, -5.22644277e-02, 9.97264689e-01], [3.20465306e-18, -5.23359562e-02, 9.98629535e-01]]
        ],
        [
            [[-5.22644277e-02, -5.22644277e-02, 9.97264689e-01], [-5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
            [[-5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
            [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [3.20465306e-18, 5.23359562e-02, 9.98629535e-01]],
            [[3.20465306e-18, 5.23359562e-02, 9.98629535e-01], [-5.22644277e-02, -5.22644277e-02, 9.97264689e-01]]
        ],
        [
            [[3.20465306e-18, 5.23359562e-02, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
            [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
            [[5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [5.22644277e-02, 5.22644277e-02, 9.97264689e-01]],
            [[5.22644277e-02, 5.22644277e-02, 9.97264689e-01], [3.20465306e-18, 5.23359562e-02, 9.98629535e-01]]
        ]
    ])

    constLat_cart = 1

    weights = _zonal_face_weights(face_edges_cart,
                                  np.array([[[1.4968158, 1.57079633],
                                           [3.14159265, 0.]],
                                          [[1.4968158, 1.57079633],
                                           [4.71238898, 3.14159265]],
                                          [[1.4968158, 1.57079633],
                                           [0., 3.14159265]],
                                          [[1.4968158, 1.57079633],
                                           [1.57079633, 4.71238898]]]),
                                  np.array([4, 4, 4, 4]),
                                  z=constLat_cart)

    expected_weights = np.array([0.25, 0.25, 0.25, 0.25])

    nt.assert_array_equal(weights, expected_weights)


def test_get_zonal_faces_weight_at_constLat_on_pole_one_face():
    # The face is touching the pole, so the weight should be 1.0 since there's only 1 face
    face_edges_cart = np.array([[
        [[-5.22644277e-02, -5.22644277e-02, -9.97264689e-01],
         [-5.23359562e-02, -6.40930613e-18, -9.98629535e-01]],
        [[-5.23359562e-02, -6.40930613e-18, -9.98629535e-01],
         [6.12323400e-17, 0.00000000e+00, -1.00000000e+00]],
        [[6.12323400e-17, 0.00000000e+00, -1.00000000e+00],
         [3.20465306e-18, -5.23359562e-02, -9.98629535e-01]],
        [[3.20465306e-18, -5.23359562e-02, -9.98629535e-01],
         [-5.22644277e-02, -5.22644277e-02, -9.97264689e-01]]
    ]])

    # Corrected face_bounds
    face_bounds = np.array([
        [-1.57079633, -1.4968158],
        [3.14159265, 0.]
    ])
    constLat_cart = -1

    weights = _zonal_face_weights(face_edges_cart,
                                  np.array([face_bounds]),
                                  np.array([4]),
                                  z=constLat_cart)

    expected_weights = np.array([1.0])

    nt.assert_array_equal(weights, expected_weights)


def test_get_zonal_faces_weight_at_constLat_on_pole_faces():
    # There will be 4 faces touching the pole, so the weight should be 0.25 for each face
    face_edges_cart = np.array([
        [
            [[5.22644277e-02, -5.22644277e-02, 9.97264689e-01], [5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
            [[5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
            [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [3.20465306e-18, -5.23359562e-02, 9.98629535e-01]],
            [[3.20465306e-18, -5.23359562e-02, 9.98629535e-01], [5.22644277e-02, -5.22644277e-02, 9.97264689e-01]]
        ],
        [
            [[3.20465306e-18, -5.23359562e-02, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
            [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [-5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
            [[-5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [-5.22644277e-02, -5.22644277e-02, 9.97264689e-01]],
            [[-5.22644277e-02, -5.22644277e-02, 9.97264689e-01], [3.20465306e-18, -5.23359562e-02, 9.98629535e-01]]
        ],
        [
            [[-5.22644277e-02, -5.22644277e-02, 9.97264689e-01], [-5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
            [[-5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
            [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [3.20465306e-18, 5.23359562e-02, 9.98629535e-01]],
            [[3.20465306e-18, 5.23359562e-02, 9.98629535e-01], [-5.22644277e-02, -5.22644277e-02, 9.97264689e-01]]
        ],
        [
            [[3.20465306e-18, 5.23359562e-02, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
            [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
            [[5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [5.22644277e-02, 5.22644277e-02, 9.97264689e-01]],
            [[5.22644277e-02, 5.22644277e-02, 9.97264689e-01], [3.20465306e-18, 5.23359562e-02, 9.98629535e-01]]
        ]
    ])

    constLat_cart = 1

    weights = _zonal_face_weights(face_edges_cart,
                                  np.array([[[1.4968158, 1.57079633],
                                           [3.14159265, 0.]],
                                          [[1.4968158, 1.57079633],
                                           [4.71238898, 3.14159265]],
                                          [[1.4968158, 1.57079633],
                                           [0., 3.14159265]],
                                          [[1.4968158, 1.57079633],
                                           [1.57079633, 4.71238898]]]),
                                  np.array([4, 4, 4, 4]),
                                  z=constLat_cart)

    expected_weights = np.array([0.25, 0.25, 0.25, 0.25])

    nt.assert_array_equal(weights, expected_weights)


def test_get_zonal_faces_weight_at_constLat_on_pole_one_face():
    # The face is touching the pole, so the weight should be 1.0 since there's only 1 face
    face_edges_cart = np.array([[
        [[-5.22644277e-02, -5.22644277e-02, -9.97264689e-01],
         [-5.23359562e-02, -6.40930613e-18, -9.98629535e-01]],
        [[-5.23359562e-02, -6.40930613e-18, -9.98629535e-01],
         [6.12323400e-17, 0.00000000e+00, -1.00000000e+00]],
        [[6.12323400e-17, 0.00000000e+00, -1.00000000e+00],
         [3.20465306e-18, -5.23359562e-02, -9.98629535e-01]],
        [[3.20465306e-18, -5.23359562e-02, -9.98629535e-01],
         [-5.22644277e-02, -5.22644277e-02, -9.97264689e-01]]
    ]])

    face_bounds = np.array([
        [-1.57079633, -1.4968158],
        [3.14159265, 0.]
    ])
    constLat_cart = -1

    weights = _zonal_face_weights(face_edges_cart,
                                  np.array([face_bounds]),
                                  np.array([4]),
                                  z=constLat_cart)

    expected_weights = np.array([1.0])

    nt.assert_array_equal(weights, expected_weights)


def test_get_zonal_faces_weight_at_constLat_on_pole_faces():
    # There will be 4 faces touching the pole, so the weight should be 0.25 for each face
    face_edges_cart = np.array([
        [
            [[5.22644277e-02, -5.22644277e-02, 9.97264689e-01], [5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
            [[5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
            [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [3.20465306e-18, -5.23359562e-02, 9.98629535e-01]],
            [[3.20465306e-18, -5.23359562e-02, 9.98629535e-01], [5.22644277e-02, -5.22644277e-02, 9.97264689e-01]]
        ],
        [
            [[3.20465306e-18, -5.23359562e-02, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
            [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [-5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
            [[-5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [-5.22644277e-02, -5.22644277e-02, 9.97264689e-01]],
            [[-5.22644277e-02, -5.22644277e-02, 9.97264689e-01], [3.20465306e-18, -5.23359562e-02, 9.98629535e-01]]
        ],
        [
            [[-5.22644277e-02, -5.22644277e-02, 9.97264689e-01], [-5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
            [[-5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
            [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [3.20465306e-18, 5.23359562e-02, 9.98629535e-01]],
            [[3.20465306e-18, 5.23359562e-02, 9.98629535e-01], [-5.22644277e-02, -5.22644277e-02, 9.97264689e-01]]
        ],
        [
            [[3.20465306e-18, 5.23359562e-02, 9.98629535e-01], [6.12323400e-17, 0.00000000e+00, 1.00000000e+00]],
            [[6.12323400e-17, 0.00000000e+00, 1.00000000e+00], [5.23359562e-02, 0.00000000e+00, 9.98629535e-01]],
            [[5.23359562e-02, 0.00000000e+00, 9.98629535e-01], [5.22644277e-02, 5.22644277e-02, 9.97264689e-01]],
            [[5.22644277e-02, 5.22644277e-02, 9.97264689e-01], [3.20465306e-18, 5.23359562e-02, 9.98629535e-01]]
        ]
    ])

    constLat_cart = 1

    weights = _zonal_face_weights(face_edges_cart,
                                  np.array([[[1.4968158, 1.57079633],
                                           [3.14159265, 0.]],
                                          [[1.4968158, 1.57079633],
                                           [4.71238898, 3.14159265]],
                                          [[1.4968158, 1.57079633],
                                           [0., 3.14159265]],
                                          [[1.4968158, 1.57079633],
                                           [1.57079633, 4.71238898]]]),
                                  np.array([4, 4, 4, 4]),
                                  z=constLat_cart)

    expected_weights = np.array([0.25, 0.25, 0.25, 0.25])

    nt.assert_array_equal(weights, expected_weights)


def test_get_zonal_faces_weight_at_constLat_latlonface():
    face_0 = [[np.deg2rad(350), np.deg2rad(40)], [np.deg2rad(350), np.deg2rad(20)],
              [np.deg2rad(10), np.deg2rad(20)], [np.deg2rad(10), np.deg2rad(40)]]
    face_1 = [[np.deg2rad(5), np.deg2rad(20)], [np.deg2rad(5), np.deg2rad(10)],
              [np.deg2rad(25), np.deg2rad(10)], [np.deg2rad(25), np.deg2rad(20)]]
    face_2 = [[np.deg2rad(30), np.deg2rad(40)], [np.deg2rad(30), np.deg2rad(20)],
              [np.deg2rad(40), np.deg2rad(20)], [np.deg2rad(40), np.deg2rad(40)]]

    # Convert the face vertices to xyz coordinates
    face_0 = [_lonlat_rad_to_xyz(*v) for v in face_0]
    face_1 = [_lonlat_rad_to_xyz(*v) for v in face_1]
    face_2 = [_lonlat_rad_to_xyz(*v) for v in face_2]

    face_0_edge_nodes = np.array([[face_0[0], face_0[1]],
                                  [face_0[1], face_0[2]],
                                  [face_0[2], face_0[3]],
                                  [face_0[3], face_0[0]]])
    face_1_edge_nodes = np.array([[face_1[0], face_1[1]],
                                  [face_1[1], face_1[2]],
                                  [face_1[2], face_1[3]],
                                  [face_1[3], face_1[0]]])
    face_2_edge_nodes = np.array([[face_2[0], face_2[1]],
                                  [face_2[1], face_2[2]],
                                  [face_2[2], face_2[3]],
                                  [face_2[3], face_2[0]]])

    face_0_latlon_bound = np.array([[np.deg2rad(20), np.deg2rad(40)],
                                    [np.deg2rad(350), np.deg2rad(10)]])
    face_1_latlon_bound = np.array([[np.deg2rad(10), np.deg2rad(20)],
                                    [np.deg2rad(5), np.deg2rad(25)]])
    face_2_latlon_bound = np.array([[np.deg2rad(20), np.deg2rad(40)],
                                    [np.deg2rad(30), np.deg2rad(40)]])

    latlon_bounds = np.array([
        face_0_latlon_bound, face_1_latlon_bound, face_2_latlon_bound
    ])

    sum = 17.5 + 17.5 + 10
    expected_weight_df = pd.DataFrame({
        "face_index": [0, 1, 2],
        "weight": [17.5 / sum, 17.5 / sum, 10 / sum]
    })

    # Assert the results is the same to the 3 decimal places
    weight_df = _zonal_face_weights_robust(np.array([
        face_0_edge_nodes, face_1_edge_nodes, face_2_edge_nodes
    ]), np.sin(np.deg2rad(20)), latlon_bounds, is_latlonface=True)

    nt.assert_array_almost_equal(weight_df, expected_weight_df, decimal=3)


def test_compare_zonal_weights(gridpath):
    """Test that the new and old zonal weight functions produce the same results."""
    uxgrid = ux.open_grid(gridpath("ugrid", "outCSne30", "outCSne30.ug"))

    # Get face edge nodes in Cartesian coordinates
    face_edge_nodes_xyz = _get_cartesian_face_edge_nodes_array(
        uxgrid.face_node_connectivity.values,
        uxgrid.node_x.values,
        uxgrid.node_y.values,
        uxgrid.node_z.values,
        uxgrid.n_max_face_nodes
    )

    n_nodes_per_face = uxgrid.n_nodes_per_face.values

    # Test at multiple latitudes
    latitudes = [-60, -30, 0, 30, 60]

    for i, lat in enumerate(latitudes):
        face_indices = uxgrid.get_faces_at_constant_latitude(lat)
        z = np.sin(np.deg2rad(lat))

        face_edge_nodes_xyz_candidate = face_edge_nodes_xyz[face_indices, :, :, :]
        n_nodes_per_face_candidate = n_nodes_per_face[face_indices]
        bounds_candidate = uxgrid.bounds.values[face_indices]

        new_weights = _zonal_face_weights(face_edge_nodes_xyz_candidate,
                                          bounds_candidate,
                                          n_nodes_per_face_candidate,
                                          z)

        existing_weights = _zonal_face_weights_robust(
            face_edge_nodes_xyz_candidate, z, bounds_candidate
        )["weight"].to_numpy()

        abs_diff = np.abs(new_weights - existing_weights)

        # For each latitude, make sure the absolute difference is below our error tolerance
        assert abs_diff.max() < ERROR_TOLERANCE
