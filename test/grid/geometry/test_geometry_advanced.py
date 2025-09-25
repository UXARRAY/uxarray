import numpy as np
import numpy.testing as nt
import pytest

import uxarray as ux
from uxarray.constants import ERROR_TOLERANCE, INT_FILL_VALUE
from uxarray.grid.coordinates import _lonlat_rad_to_xyz, _normalize_xyz, _xyz_to_lonlat_rad
from uxarray.grid.arcs import extreme_gca_z
from uxarray.grid.bounds import _get_latlonbox_width, insert_pt_in_latlonbox
from uxarray.grid.geometry import _pole_point_inside_polygon_cartesian


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
    v_temp = np.cross(v1_cart, v2_cart)
    v0 = np.cross(v_temp, v1_cart)
    v0 = _normalize_xyz(*v0.tolist())
    max_section = [v1_cart, v2_cart]

    # Iteratively find the maximum latitude
    while np.abs(b_lonlat[1] - c_lonlat[1]) >= ERROR_TOLERANCE or np.abs(
            b_lonlat[0] - c_lonlat[0]) >= ERROR_TOLERANCE:
        max_lat = -np.pi
        v_b, v_c = max_section
        angle_v1_v2_rad = ux.grid.arcs._angle_of_2_vectors(v_b, v_c)
        v0 = np.cross(v_temp, v_b)
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
        b_lonlat = _xyz_to_lonlat_rad(*max_section[0].tolist())
        c_lonlat = _xyz_to_lonlat_rad(*max_section[1].tolist())

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
    v_temp = np.cross(v1_cart, v2_cart)
    v0 = np.cross(v_temp, v1_cart)
    v0 = np.array(_normalize_xyz(*v0.tolist()))
    min_section = [v1_cart, v2_cart]

    # Iteratively find the minimum latitude
    while np.abs(b_lonlat[1] - c_lonlat[1]) >= ERROR_TOLERANCE or np.abs(
            b_lonlat[0] - c_lonlat[0]) >= ERROR_TOLERANCE:
        min_lat = np.pi
        v_b, v_c = min_section
        angle_v1_v2_rad = ux.grid.arcs._angle_of_2_vectors(v_b, v_c)
        v0 = np.cross(v_temp, v_b)
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
        b_lonlat = _xyz_to_lonlat_rad(*min_section[0].tolist())
        c_lonlat = _xyz_to_lonlat_rad(*min_section[1].tolist())

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
