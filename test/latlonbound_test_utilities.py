"""
This file stores the necessary helper functions for the latlonbox test cases.
"""
import copy

import numpy as np

import uxarray.helpers


def max_latitude_rad(v1, v2):
    """Quantitative method to find the maximum latitude between in a great circle arc

    Parameters
    ----------
    v1: float array
        The first endpoint of the great circle arc [lon, lat] in degree east.
    v2: float array
        The second endpoint of the great circle arc [lon, lat] in degree east.

    Returns
    -------
    float
        maximum latitude in radian
    """

    # Find the parametrized equation for the great circle passing through v1 and v2
    err_tolerance = 1.0e-15
    b_lonlat = np.deg2rad(v1)
    c_lonlat = np.deg2rad(v2)

    v1_cart = uxarray.helpers._convert_node_lonlat_rad_to_xyz(np.deg2rad(v1))
    v2_cart = uxarray.helpers._convert_node_lonlat_rad_to_xyz(np.deg2rad(v2))

    max_section = np.array([v1_cart, v2_cart])  # record the subsection that has the maximum latitude

    # Only stop the iteration when two endpoints are extremely closed
    while np.absolute(b_lonlat[1] - c_lonlat[1]) >= err_tolerance or np.absolute(
            b_lonlat[0] - c_lonlat[0]) >= err_tolerance:
        max_lat = -np.pi  # reset the max_latitude for each while loop
        v_b = max_section[0]
        v_c = max_section[1]

        # Divide the angle of v1/v2 into 10 subsections, the leftover will be put in the last one
        # Update v0 based on max_section[0], since the angle is always from max_section[0] to v0
        angle_v1_v2_rad = uxarray.helpers.angle_of_2_vectors(v_b, v_c)
        v0 = np.array(uxarray.helpers._normalize_in_place(np.cross(np.cross(v1_cart, v2_cart), v_b)))
        avg_angle_rad = angle_v1_v2_rad / 10

        for i in range(0, 10):
            angle_rad_prev = avg_angle_rad * i
            if i >= 9:
                angle_rad_next = angle_v1_v2_rad
            else:
                angle_rad_next = angle_rad_prev + avg_angle_rad

            # Get the two vectors of this section
            w1_new = np.cos(angle_rad_prev) * v_b + np.sin(angle_rad_prev) * v0
            w2_new = np.cos(angle_rad_next) * v_b + np.sin(angle_rad_next) * v0

            # convert the 3D [x, y, z] vector into 2D lat/lon vector
            w1_lonlat = uxarray.helpers._convert_node_xyz_to_lonlat_rad(list(w1_new))
            w2_lonlat = uxarray.helpers._convert_node_xyz_to_lonlat_rad(list(w2_new))

            # Manually set the left and right boundaries to avoid error accumulation
            if i == 0:
                w1_lonlat[1] = b_lonlat[1]
            elif i >= 9:
                w2_lonlat[1] = c_lonlat[1]

            max_lat = max(max_lat, w1_lonlat[1], w2_lonlat[1])

            if np.absolute(w2_lonlat[1] -
                           w1_lonlat[1]) <= err_tolerance or w1_lonlat[
                1] == max_lat == w2_lonlat[1]:
                max_section = np.array([w1_new, w2_new])
                break

            # if the largest absolute value of lat at each sub-interval point b_i.
            # Repeat algorithm with the sub-interval points (b,c)=(b_{i-1},b_{i+1})
            if np.absolute(max_lat - w1_lonlat[1]) <= err_tolerance:
                if i != 0:
                    angle_rad_prev -= avg_angle_rad
                    w1_new = np.cos(angle_rad_prev) * v_b + np.sin(angle_rad_prev) * v0
                    w2_new = np.cos(angle_rad_next) * v_b + np.sin(angle_rad_next) * v0
                    max_section = [w1_new, w2_new]
                else:
                    max_section = [v_b, w2_new]

            elif np.absolute(max_lat - w2_lonlat[1]) <= err_tolerance:
                if i != 9:
                    angle_rad_next += avg_angle_rad
                    w1_new = np.cos(angle_rad_prev) * v_b+ np.sin(angle_rad_prev) * v0
                    w2_new = np.cos(angle_rad_next) * v_b + np.sin(angle_rad_next) * v0
                    max_section = [w1_new, w2_new]
                else:
                    max_section = [w1_new, v_c]

        b_lonlat = uxarray.helpers._convert_node_xyz_to_lonlat_rad(copy.deepcopy(max_section[0]))
        c_lonlat = uxarray.helpers._convert_node_xyz_to_lonlat_rad(copy.deepcopy(max_section[1]))

    return np.average([b_lonlat[1], c_lonlat[1]])


def min_latitude_rad(v1, v2):
    """Quantitative method to find the minimum latitude between in a great circle arc recursively

    Parameters
    ----------
    v1: float array
        The first endpoint of the great circle arc [lon, lat] in degree east.
    v2: float array
        The second endpoint of the great circle arc [lon, lat] in degree east.

    Returns
    -------
    float
        minimum latitude in radian
    """

    # Find the parametrized equation for the great circle passing through v1 and v2
    err_tolerance = 1.0e-15
    b_lonlat = np.deg2rad(v1)
    c_lonlat = np.deg2rad(v2)

    v1_cart = uxarray.helpers._convert_node_lonlat_rad_to_xyz(np.deg2rad(v1))
    v2_cart = uxarray.helpers._convert_node_lonlat_rad_to_xyz(np.deg2rad(v2))
    v_temp = np.cross(v1_cart, v2_cart)
    v0 = np.cross(v_temp, v1_cart)
    v0 = uxarray.helpers._normalize_in_place(v0)

    min_section = [v1_cart,
                   v2_cart]  # record the subsection that has the maximum latitude

    # Only stop the iteration when two endpoints are extremely closed
    while np.absolute(b_lonlat[1] - c_lonlat[1]) >= err_tolerance or np.absolute(
            b_lonlat[0] - c_lonlat[0]) >= err_tolerance:
        min_lat = np.pi  # reset the max_latitude for each while loop
        v_b = min_section[0]
        v_c = min_section[1]

        # Divide the angle of v1/v2 into 10 subsections, the leftover will be put in the last one
        # Update v0 based on min_section[0], since the angle is always from min_section[0] to v0
        angle_v1_v2_rad = uxarray.helpers.angle_of_2_vectors(v_b, v_c)
        v0 = np.cross(v_temp, v_b)
        v0 = uxarray.helpers._normalize_in_place(v0)
        avg_angle_rad = angle_v1_v2_rad / 10

        for i in range(0, 10):
            angle_rad_prev = avg_angle_rad * i
            if i >= 9:
                angle_rad_next = angle_v1_v2_rad
            else:
                angle_rad_next = angle_rad_prev + avg_angle_rad

            # Get the two vectors of this section
            w1_new = [np.cos(angle_rad_prev) * v_b[i] + np.sin(
                angle_rad_prev) * v0[i] for i in range(0, len(v_b))]
            w2_new = [np.cos(angle_rad_next) * v_b[i] + np.sin(
                angle_rad_next) * v0[i] for i in range(0, len(v_b))]

            # convert the 3D [x, y, z] vector into 2D lat/lon vector
            w1_lonlat = uxarray.helpers._convert_node_xyz_to_lonlat_rad(w1_new)
            w2_lonlat = uxarray.helpers._convert_node_xyz_to_lonlat_rad(w2_new)

            # Manually set the left and right boundaries to avoid error accumulation
            if i == 0:
                w1_lonlat[1] = b_lonlat[1]
            elif i >= 9:
                w2_lonlat[1] = c_lonlat[1]

            min_lat = min(min_lat, w1_lonlat[1], w2_lonlat[1])

            if np.absolute(w2_lonlat[1] -
                           w1_lonlat[1]) <= err_tolerance or w1_lonlat[
                1] == min_lat == w2_lonlat[1]:
                min_section = [w1_new, w2_new]
                break

            # if the largest absolute value of lat at each sub-interval point b_i.
            # Repeat algorithm with the sub-interval points (b,c)=(b_{i-1},b_{i+1})
            if np.absolute(min_lat - w1_lonlat[1]) <= err_tolerance:
                if i != 0:
                    angle_rad_prev -= avg_angle_rad
                    w1_new = [np.cos(angle_rad_prev) * v_b[i] + np.sin(
                        angle_rad_prev) * v0[i] for i in range(0, len(v_b))]
                    w2_new = [np.cos(angle_rad_next) * v_b[i] + np.sin(
                        angle_rad_next) * v0[i] for i in range(0, len(v_b))]
                    min_section = [w1_new, w2_new]
                else:
                    min_section = [v_b, w2_new]

            elif np.absolute(min_lat - w2_lonlat[1]) <= err_tolerance:
                if i != 9:
                    angle_rad_next += avg_angle_rad
                    w1_new = [np.cos(angle_rad_prev) * v_b[i] + np.sin(
                        angle_rad_prev) * v0[i] for i in range(0, len(v_b))]
                    w2_new = [np.cos(angle_rad_next) * v_b[i] + np.sin(
                        angle_rad_next) * v0[i] for i in range(0, len(v_b))]
                    min_section = [w1_new, w2_new]
                else:
                    min_section = [w1_new, v_c]

        b_lonlat = uxarray.helpers._convert_node_xyz_to_lonlat_rad(copy.deepcopy(min_section[0]))
        c_lonlat = uxarray.helpers._convert_node_xyz_to_lonlat_rad(copy.deepcopy(min_section[1]))

    return np.average([b_lonlat[1], c_lonlat[1]])


def minmax_Longitude_rad(v1, v2):
    """Quantitative method to find the minimum and maximum Longitude between in a great circle

    Parameters
    ----------
    v1: float array
        The first endpoint of the great circle arc [lon, lat] in degree east.
    v1: float array
        The second endpoint of the great circle arc [lon, lat] in degree east.

    Returns
    -------
    float array
        [lon_min, lon_max] in radian
    """
    # First reorder the two ends points based on the rule: the span of its longitude must less than 180 degree
    [start_lon, end_lon] = np.sort([v1[0], v2[0]])
    if end_lon - start_lon <= 180:
        return [np.deg2rad(start_lon), np.deg2rad(end_lon)]
    else:
        # swap the start and end longitude
        temp_lon = start_lon
        start_lon = end_lon
        end_lon = temp_lon
    return [np.deg2rad(start_lon), np.deg2rad(end_lon)]

# Helper function for the test_generate_Latlon_bounds_longitude_minmax
def expand_longitude_rad(min_lon_rad_edge, max_lon_rad_edge, minmax_lon_rad_face):
    """Helper function top expand the longitude boundary of a face
    Parameters
    ----------
    min_lon_rad_edge, max_lon_rad_edge: float
    minmax_lon_rad_face: float array [min_lon_rad_face, max_lon_rad_face]
    Returns:
    minmax_lon_rad_face: float array [new_min_lon_rad_face, new_max_lon_rad_face]
    """
    # Longitude range expansion: Compare between [min_lon_rad_edge, max_lon_rad_edge] and minmax_lon_rad_face
    if minmax_lon_rad_face[0] <= minmax_lon_rad_face[1]:
        if min_lon_rad_edge <= max_lon_rad_edge:
            if min_lon_rad_edge < minmax_lon_rad_face[0] and max_lon_rad_edge < minmax_lon_rad_face[1]:
                # First try to add from the left:
                left_width = minmax_lon_rad_face[1] - min_lon_rad_edge
                if left_width <= np.pi:
                    minmax_lon_rad_face = [min_lon_rad_edge, minmax_lon_rad_face[1]]
                else:
                    # add from the right:
                    minmax_lon_rad_face = [minmax_lon_rad_face[0], min_lon_rad_edge]

            elif min_lon_rad_edge > minmax_lon_rad_face[0] and max_lon_rad_edge > minmax_lon_rad_face[1]:
                # First try to add from the right
                right_width = max_lon_rad_edge - minmax_lon_rad_face[0]
                if right_width <= np.pi:
                    minmax_lon_rad_face = [minmax_lon_rad_face[0], max_lon_rad_edge]
                else:
                    # then add from the left
                    minmax_lon_rad_face = [max_lon_rad_edge, minmax_lon_rad_face[1]]

            else:
                minmax_lon_rad_face = [min(min_lon_rad_edge, minmax_lon_rad_face[0]),
                                       max(max_lon_rad_edge, minmax_lon_rad_face[1])]

        else:
            # The min_lon_rad_edge is on the left side of minmax_lon_rad_face range
            if minmax_lon_rad_face[1] <= np.pi:
                minmax_lon_rad_face = [min_lon_rad_edge, max(max_lon_rad_edge, minmax_lon_rad_face[1])]
            else:
                # if it's on the right side of the minmax_lon_rad_face range
                minmax_lon_rad_face = [min(min_lon_rad_edge, minmax_lon_rad_face[0]), max_lon_rad_edge]

    else:
        if min_lon_rad_edge <= max_lon_rad_edge:
            if __on_left(minmax_lon_rad_face, [min_lon_rad_edge, max_lon_rad_edge], safe_call=True):
                # First try adding from the left:
                left_width = (2 * np.pi - min_lon_rad_edge) + minmax_lon_rad_face[1]
                if left_width <= np.pi:
                    minmax_lon_rad_face = [min_lon_rad_edge, minmax_lon_rad_face[1]]
                else:
                    # Then add from the right
                    minmax_lon_rad_face = [minmax_lon_rad_face[0], min_lon_rad_edge]

            elif __on_right(minmax_lon_rad_face, [min_lon_rad_edge, max_lon_rad_edge], safe_call=True):
                # First try adding from the right
                right_width = (2 * np.pi - minmax_lon_rad_face[0]) + max_lon_rad_edge
                if right_width <= np.pi:
                    minmax_lon_rad_face = [minmax_lon_rad_face[0], max_lon_rad_edge]
                else:
                    # Then try adding from the left
                    minmax_lon_rad_face = [max_lon_rad_edge, minmax_lon_rad_face[1]]

            else:
                if within(minmax_lon_rad_face[1], min_lon_rad_edge, minmax_lon_rad_face[0]):
                    minmax_lon_rad_face[0] = min_lon_rad_edge
                else:
                    minmax_lon_rad_face[0] = minmax_lon_rad_face[0]

                if 2 * np.pi > max_lon_rad_edge >= minmax_lon_rad_face[0] or max_lon_rad_edge < minmax_lon_rad_face[1]:
                    minmax_lon_rad_face[1] = minmax_lon_rad_face[1]
                else:
                    minmax_lon_rad_face[1] = max(minmax_lon_rad_face[1], max_lon_rad_edge)

        else:
            minmax_lon_rad_face[0] = min(min_lon_rad_edge, minmax_lon_rad_face[0])
            minmax_lon_rad_face[1] = max(max_lon_rad_edge, minmax_lon_rad_face[1])

    return minmax_lon_rad_face


# helper function to determine whether the insert_edge is on the left side of the ref_edge
def __on_left(ref_edge, insert_edge, safe_call=False):
    """Helper function used for the longitude test case only. Only designed to consider a specific scenario
    as described below
    Parameters
    ----------
    ref_edge: The edge that goes across the 0 longitude line: [min_longitude, max_longitude] and min_long > max_long
    insert_edge: the inserted edge, [min_longitude, max_longitude]
    safe_call (default to be False): When call this function, user must make sure it's under the safe and ideal condition
    Returns: boolean
    True: the insert_edge is on the left side of the ref_edge ( the insert_edge's min_longitude
            is larger than 180 longitude, and its max_longitude between 180 longitude and the max_longitude of the ref_edge
    False: It's not on the left side of the ref_edge. Cannot guarantee it's on the right side
    """
    if ref_edge[0] <= ref_edge[1]:
        raise Exception('This function can only be applied to the edge that goes across the 0 longitude line')
    if not safe_call:
        raise Exception('Calling this function here is not safe')
    left_flag = False
    if insert_edge[1] >= ref_edge[1] and insert_edge[1] >= ref_edge[0]:
        if _within(ref_edge[1], insert_edge[0], ref_edge[0]):
            left_flag = True
    elif insert_edge[1] <= ref_edge[1] and insert_edge[1] <= ref_edge[0]:
        if _within(ref_edge[1], insert_edge[0], ref_edge[0]):
            left_flag = True
    return left_flag


# helper function to determine whether the insert_edge is on the right side of the ref_edge
def __on_right(ref_edge, insert_edge, safe_call=False):
    """Helper function used for the longitude test case only. Only designed to consider a specific scenario
    as described below
    Parameters
    ----------
    ref_edge: The edge that goes across the 0 longitude line: [min_longitude, max_longitude] and min_long > max_long
    insert_edge: the inserted edge, [min_longitude, max_longitude]
    safe_call (default to be False): When call this function, user must make sure it's under the safe and ideal condition
    Returns: boolean
    True: the insert_edge is on the right side of the ref_edge ( the insert_edge's min_longitude
            is between the ref_edge's min_longitude and 0 longitude, and the insert_edge's max_longitude is between
            ref_edge's max_longitude and 180 longitude
    False: It's not on the right side of the ref_edge. Cannot guarantee it's on the left side
    """
    if ref_edge[0] <= ref_edge[1]:
        raise Exception('This function can only be applied to the edge that goes across the 0 longitude line')
    if not safe_call:
        raise Exception('Calling this function here is not safe')
    right_flag = False
    if insert_edge[0] >= ref_edge[0] and insert_edge[0] >= ref_edge[1]:
        if _within(ref_edge[1], insert_edge[1], ref_edge[0]):
            right_flag = True
    elif insert_edge[0] <= ref_edge[0] and insert_edge[0] <= ref_edge[1]:
        if _within(ref_edge[1], insert_edge[1], ref_edge[0]):
            right_flag = True

    return right_flag