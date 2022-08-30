import numpy as np
import copy

from .helpers import angle_of_2_vectors, within, normalize_in_place, convert_node_xyz_to_lonlat_rad, convert_node_lonlat_rad_to_xyz

# helper function to insert a new point into the latlon box
def insert_pt_in_latlonbox(old_box, new_pt, is_lon_periodic=True):
    """Compare the new point's latitude and longitude with the target the
    latlonbox.
    Parameters: old_box: float array, the original lat lon box [[lat_0, lat_1],[lon_0, lon_1]],required
                new_pt: float array, the new lat lon point [lon, lat], required
                is_lon_periodic: Flag indicating the latlonbox is a regional (default to be True).
    Returns: float array, a lat lon box [[lat_0, lat_1],[lon_0, lon_1]]
    Raises:
       Exception: Logic Errors
    """
    # If the box is null (no point inserted yet)

    if old_box[0][0] == old_box[0][1] == 404.0:
        latlon_box = old_box
        latlon_box[0] = [new_pt[0], new_pt[0]]

    if old_box[1][0] == old_box[1][1] == 404.0:
        latlon_box = old_box
        latlon_box[1] = [new_pt[1], new_pt[1]]

    if old_box[0][0] == old_box[0][1] == old_box[1][0] == old_box[1][1] == 404.0:
        return latlon_box

    # Deal with the pole point
    if new_pt[1] == 404.0 and (
            (np.absolute(new_pt[0] - 0.5 * np.pi) < 1.0e-12) or (np.absolute(new_pt[0] - (-0.5 * np.pi)) < 1.0e-12)):
        latlon_box = old_box
        if np.absolute(new_pt[0] - 0.5 * np.pi) < 1.0e-12:
            latlon_box[0][1] = 0.5 * np.pi
        elif np.absolute(new_pt[0] - (-0.5 * np.pi)) < 1.0e-12:
            latlon_box[0][0] = -0.5 * np.pi
        return latlon_box

    old_lon_width = 2.0 * np.pi
    lat_pt = new_pt[0]
    lon_pt = new_pt[1]
    latlon_box = old_box  # The returned box

    if lon_pt < 0.0:
        raise Exception('lon_pt out of range ( {} < 0)"'.format(lon_pt))

    if lon_pt > old_lon_width:
        raise Exception('lon_pt out of range ( {} > {})"'.format(
            lon_pt, old_lon_width))

    # Expand latitudes
    if lat_pt > latlon_box[0][1]:
        latlon_box[0][1] = lat_pt

    if lat_pt < latlon_box[0][0]:
        latlon_box[0][0] = lat_pt

    # Expand longitude, if non-periodic
    if not is_lon_periodic:
        if lon_pt > latlon_box[1][1]:
            latlon_box[1][1] = lon_pt
        if lon_pt < latlon_box[1][0]:
            latlon_box[1][0] = lon_pt
        return latlon_box

    # New longitude lies within existing range
    if latlon_box[1][0] <= latlon_box[1][1]:
        if lon_pt >= latlon_box[1][0] and lon_pt <= latlon_box[1][1]:
            return latlon_box
    else:
        if lon_pt >= latlon_box[1][0] or lon_pt <= latlon_box[1][1]:
            return latlon_box

    # New longitude lies outside of existing range
    box_a = copy.deepcopy(latlon_box)
    box_a[1][0] = lon_pt

    box_b = copy.deepcopy(latlon_box)
    box_b[1][1] = lon_pt

    # The updated box is the box of minimum width
    d_width_now = get_latlonbox_width(latlon_box)
    d_width_a = get_latlonbox_width(box_a)
    d_width_b = get_latlonbox_width(box_b)

    if (d_width_a - d_width_now) < -1.0e-14 or (d_width_b -
                                                d_width_now) < -1.0e-14:
        raise Exception('logic error')

    if d_width_a < d_width_b:
        return box_a
    else:
        return box_b


# helper function to calculate the latlonbox width
def get_latlonbox_width(latlonbox, is_lon_periodic=True):
    """Calculate the width of this LatLonBox
    Parameters: latlonbox: float array, lat lon box [[lat_0, lat_1],[lon_0, lon_1]],required
                is_lon_periodic: boolean, Flag indicating the latlonbox is a regional (default to be True).
    Returns: float array, a lat lon box [[lat_0, lat_1],[lon_0, lon_1]]
    Raises:
       Exception: Logic Errors
    """

    if not is_lon_periodic:
        return latlonbox[1][1] - latlonbox[1][0]

    if latlonbox[1][0] == latlonbox[1][1]:
        return 0.0
    elif latlonbox[1][0] <= latlonbox[1][1]:
        return latlonbox[1][1] - latlonbox[1][0]
    else:
        return latlonbox[1][1] - latlonbox[1][0] + (2 * np.pi)


# Quantitative method to find the maximum latitude between in a great circle arc
def max_latitude_rad(v1, v2):
    """Quantitative method to find the maximum latitude between in a great circle arc
    Parameters:
        v1: float array [lon, lat] in degree east
        v2: float array [lon, lat] in degree east
    Returns: float, maximum latitude in radian
    """

    # Find the parametrized equation for the great circle passing through v1 and v2
    err_tolerance = 1.0e-15
    b_lonlat = np.deg2rad(v1)
    c_lonlat = np.deg2rad(v2)

    v1_cart = convert_node_lonlat_rad_to_xyz(np.deg2rad(v1))
    v2_cart = convert_node_lonlat_rad_to_xyz(np.deg2rad(v2))
    v_temp = np.cross(v1_cart, v2_cart)
    v0 = np.cross(v_temp, v1_cart)
    v0 = normalize_in_place(v0)

    max_section = [v1_cart,
                   v2_cart]  # record the subsection that has the maximum latitude

    # Only stop the iteration when two endpoints are extremely closed
    while np.absolute(b_lonlat[1] - c_lonlat[1]) >= err_tolerance or np.absolute(
            b_lonlat[0] - c_lonlat[0]) >= err_tolerance:
        max_lat = -np.pi  # reset the max_latitude for each while loop
        v_b = max_section[0]
        v_c = max_section[1]

        # Divide the angle of v1/v2 into 10 subsections, the leftover will be put in the last one
        # Update v0 based on max_section[0], since the angle is always from max_section[0] to v0
        angle_v1_v2_rad = angle_of_2_vectors(v_b, v_c)
        v0 = np.cross(v_temp, v_b)
        v0 = normalize_in_place(v0)
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
            w1_lonlat = convert_node_xyz_to_lonlat_rad(w1_new)
            w2_lonlat = convert_node_xyz_to_lonlat_rad(w2_new)

            # Manually set the left and right boundaries to avoid error accumulation
            if i == 0:
                w1_lonlat[1] = b_lonlat[1]
            elif i >= 9:
                w2_lonlat[1] = c_lonlat[1]

            max_lat = max(max_lat, w1_lonlat[1], w2_lonlat[1])

            if np.absolute(w2_lonlat[1] -
                           w1_lonlat[1]) <= err_tolerance or w1_lonlat[
                1] == max_lat == w2_lonlat[1]:
                max_section = [w1_new, w2_new]
                break

            # if the largest absolute value of lat at each sub-interval point b_i.
            # Repeat algorithm with the sub-interval points (b,c)=(b_{i-1},b_{i+1})
            if np.absolute(max_lat - w1_lonlat[1]) <= err_tolerance:
                if i != 0:
                    angle_rad_prev -= avg_angle_rad
                    w1_new = [np.cos(angle_rad_prev) * v_b[i] + np.sin(
                        angle_rad_prev) * v0[i] for i in range(0, len(v_b))]
                    w2_new = [np.cos(angle_rad_next) * v_b[i] + np.sin(
                        angle_rad_next) * v0[i] for i in range(0, len(v_b))]
                    max_section = [w1_new, w2_new]
                else:
                    max_section = [v_b, w2_new]

            elif np.absolute(max_lat - w2_lonlat[1]) <= err_tolerance:
                if i != 9:
                    angle_rad_next += avg_angle_rad
                    w1_new = [np.cos(angle_rad_prev) * v_b[i] + np.sin(
                        angle_rad_prev) * v0[i] for i in range(0, len(v_b))]
                    w2_new = [np.cos(angle_rad_next) * v_b[i] + np.sin(
                        angle_rad_next) * v0[i] for i in range(0, len(v_b))]
                    max_section = [w1_new, w2_new]
                else:
                    max_section = [w1_new, v_c]

        b_lonlat = convert_node_xyz_to_lonlat_rad(copy.deepcopy(max_section[0]))
        c_lonlat = convert_node_xyz_to_lonlat_rad(copy.deepcopy(max_section[1]))

    return np.average([b_lonlat[1], c_lonlat[1]])


# Quantitative method to find the minimum latitude between in a great circle arc recursively
def min_latitude_rad(v1, v2):
    """Quantitative method to find the minimum latitude between in a great circle arc recursively
    Parameters:
        v1: float array [lon, lat] in degree east
        v2: float array [lon, lat] in degree east
    Returns: float, minimum latitude in radian
    """

    # Find the parametrized equation for the great circle passing through v1 and v2
    err_tolerance = 1.0e-15
    b_lonlat = np.deg2rad(v1)
    c_lonlat = np.deg2rad(v2)

    v1_cart = convert_node_lonlat_rad_to_xyz(np.deg2rad(v1))
    v2_cart = convert_node_lonlat_rad_to_xyz(np.deg2rad(v2))
    v_temp = np.cross(v1_cart, v2_cart)
    v0 = np.cross(v_temp, v1_cart)
    v0 = normalize_in_place(v0)

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
        angle_v1_v2_rad = angle_of_2_vectors(v_b, v_c)
        v0 = np.cross(v_temp, v_b)
        v0 = normalize_in_place(v0)
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
            w1_lonlat = convert_node_xyz_to_lonlat_rad(w1_new)
            w2_lonlat = convert_node_xyz_to_lonlat_rad(w2_new)

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

        b_lonlat = convert_node_xyz_to_lonlat_rad(copy.deepcopy(min_section[0]))
        c_lonlat = convert_node_xyz_to_lonlat_rad(copy.deepcopy(min_section[1]))

    return np.average([b_lonlat[1], c_lonlat[1]])


# Quantitative method to find the minimum and maximum Longitude between in a great circle
def minmax_Longitude_rad(v1, v2):
    """Quantitative method to find the minimum Longitude between in a great circle arc.
      And it assumes that an edge's longitude span cannot be larger than 180 degree.
    Parameters:
        v1: float array [lon, lat] in degree east
        v2: float array [lon, lat] in degree east
    Returns: float array, [lon_min, lon_max] in radian
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


# helper function to calculate the point position of the intersection of two great circle arcs
def get_intersection_point_gcr_gcr(w0, w1, v0, v1):
    """Helper function to calculate the intersection point of two great circle
    arcs in 3D coordinates.
    Parameters
    ----------
    w0: float array [x, y, z], the end point of great circle arc w
    w1: float array [x, y, z], the other end point of great circle arc w
    v0: float array [x, y, z], the end point of great circle arc v
    v1: float array [x, y, z], the other end point of great circle arc v
    Returns: float array, the result vector [x, y, z]
     [x, y, z]: the 3D coordinates of the intersection point
     [0, 0, 0]: Indication that two great circle arcs are parallel to each other
     [-1, -1, -1]: Indication that two great circle arcs doesn't have intersection
    """
    w0 = normalize_in_place(w0)
    w1 = normalize_in_place(w1)
    v0 = normalize_in_place(v0)
    v1 = normalize_in_place(v1)
    x1 = np.cross(np.cross(w0, w1), np.cross(v0, v1)).tolist()
    x2 = [-x1[0], -x1[1], -x1[2]]

    # Find out whether X1 or X2 is within the interval [wo, w1]

    if within(w0[0], x1[0], w1[0]) and within(w0[1], x1[1], w1[1]) and within(
            w0[2], x1[2], w1[2]):
        return x1
    elif within(w0[0], x2[0], w1[0]) and within(w0[1], x2[1], w1[1]) and within(
            w0[2], x2[2], w1[2]):
        return x2
    elif x1[0] == 0 and x1[1] == 0 and x1[2] == 0:
        return [0, 0, 0]  # two vectors are parallel to each other
    else:
        return [-1, -1, -1]  # Intersection out of the interval or


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
        if within(ref_edge[1], insert_edge[0], ref_edge[0]):
            left_flag = True
    elif insert_edge[1] <= ref_edge[1] and insert_edge[1] <= ref_edge[0]:
        if within(ref_edge[1], insert_edge[0], ref_edge[0]):
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
        if within(ref_edge[1], insert_edge[1], ref_edge[0]):
            right_flag = True
    elif insert_edge[0] <= ref_edge[0] and insert_edge[0] <= ref_edge[1]:
        if within(ref_edge[1], insert_edge[1], ref_edge[0]):
            right_flag = True

    return right_flag
