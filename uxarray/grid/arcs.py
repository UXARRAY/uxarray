import math

import numpy as np
from numba import njit

from uxarray.constants import ERROR_TOLERANCE, MACHINE_EPSILON
from uxarray.grid.coordinates import (
    _normalize_xyz_scalar,
)
from uxarray.grid.utils import _angle_of_2_vectors


def _to_list(obj):
    if not isinstance(obj, list):
        if isinstance(obj, np.ndarray):
            # Convert the NumPy array to a list using .tolist()
            obj = obj.tolist()
        else:
            # If not a list or NumPy array, return the object as-is
            obj = [obj]
    return obj


@njit(cache=True)
def point_within_gca(pt_xyz, gca_a_xyz, gca_b_xyz):
    """
    Check if a point lies on a given Great Circle Arc (GCA) interval, considering the smaller arc of the circle.
    Handles the anti-meridian case as well.

    Parameters
    ----------
    pt_xyz : numpy.ndarray
        Cartesian coordinates of the point.
    gca_a_xyz : numpy.ndarray
        Cartesian coordinates of the first endpoint of the Great Circle Arc.
    gca_b_xyz : numpy.ndarray
        Cartesian coordinates of the second endpoint of the Great Circle Arc.

    Returns
    -------
    bool
        True if the point lies within the specified GCA interval, False otherwise.

    Raises
    ------
    ValueError
        If the input GCA spans exactly 180 degrees (π radians), as this GCA can have multiple planes.
        In such cases, consider breaking the GCA into two separate arcs.

    Notes
    -----
    - The function ensures that the point lies on the same plane as the GCA before performing interval checks.
    - It assumes the input represents the smaller arc of the Great Circle.
    - The `_angle_of_2_vectors` and `_xyz_to_lonlat_rad_scalar` functions are used for calculations.
    """
    # 1. Check if the input GCA spans exactly 180 degrees
    angle_ab = _angle_of_2_vectors(gca_a_xyz, gca_b_xyz)
    if np.allclose(angle_ab, np.pi, rtol=0.0, atol=MACHINE_EPSILON):
        raise ValueError(
            "The input Great Circle Arc spans exactly 180 degrees, which can correspond to multiple planes. "
            "Consider breaking the Great Circle Arc into two smaller arcs."
        )

    # 2. Verify if the point lies on the plane of the GCA
    cross_product = np.cross(gca_a_xyz, gca_b_xyz)
    if not np.allclose(
        np.dot(cross_product, pt_xyz), 0, rtol=MACHINE_EPSILON, atol=MACHINE_EPSILON
    ):
        return False

    # 3. Check if the point lies within the Great Circle Arc interval
    pt_a = gca_a_xyz - pt_xyz
    pt_b = gca_b_xyz - pt_xyz

    # Use the dot product to determine the sign of the angle between pt_a and pt_b
    cos_theta = np.dot(pt_a, pt_b)

    # Return True if the point lies within the interval (smaller arc)
    if cos_theta < 0:
        return True
    elif np.isclose(cos_theta, 0.0, atol=MACHINE_EPSILON):
        # set error tolerance to 0.0
        return True
    else:
        return False


@njit(cache=True)
def in_between(p, q, r) -> bool:
    """Determines whether the number q is between p and r.

    Parameters
    ----------
    p : float
        The lower bound.
    q : float
        The number to check.
    r : float
        The upper bound.

    Returns
    -------
    bool
        True if q is between p and r, False otherwise.
    """

    return p <= q <= r or r <= q <= p


@njit(cache=True)
def _decide_pole_latitude(lat1, lat2):
    """Determine the pole latitude based on the latitudes of two points on a
    Great Circle Arc (GCA).

    This function calculates the combined latitude span from each point to its nearest pole
    and decides which pole (North or South) the smaller GCA will pass. This decision is crucial
    for handling GCAs that span exactly or more than 180 degrees in longitude, indicating
    the arc might pass through or close to one of the Earth's poles.

    Parameters
    ----------
    lat1 : float
        Latitude of the first point in radians. Positive for the Northern Hemisphere, negative for the Southern.
    lat2 : float
        Latitude of the second point in radians. Positive for the Northern Hemisphere, negative for the Southern.

    Returns
    -------
    float
        The latitude of the pole (np.pi/2 for the North Pole or -np.pi/2 for the South Pole) the GCA is closer to.

    Notes
    -----
    The function assumes the input latitudes are valid (i.e., between -np.pi/2 and np.pi/2) and expressed in radians.
    The determination of which pole a GCA is closer to is based on the sum of the latitudinal spans from each point
    to its nearest pole, considering the shortest path on the sphere.
    """
    # Calculate the total latitudinal span to the nearest poles
    lat_extend = abs(np.pi / 2 - abs(lat1)) + np.pi / 2 + abs(lat2)

    # Determine the closest pole based on the latitudinal span
    if lat_extend < np.pi:
        closest_pole = np.pi / 2 if lat1 > 0 else -np.pi / 2
    else:
        closest_pole = -np.pi / 2 if lat1 > 0 else np.pi / 2

    return closest_pole


@njit(cache=True)
def max3(a, b, c):
    if a >= b and a >= c:
        return a
    elif b >= c:
        return b
    else:
        return c


@njit(cache=True)
def min3(a, b, c):
    if a <= b and a <= c:
        return a
    elif b <= c:
        return b
    else:
        return c


@njit(cache=True)
def clip_scalar(a, a_min, a_max):
    if a < a_min:
        return a_min
    elif a > a_max:
        return a_max
    else:
        return a


@njit(cache=True)
def extreme_gca_latitude(gca_cart, gca_lonlat, extreme_type):
    """
    Calculate the maximum or minimum latitude of a great circle arc defined
    by two 3D points.

    Parameters
    ----------
    gca_cart : numpy.ndarray
        An array containing two 3D vectors that define a great circle arc.

    gca_lonlat : numpy.ndarray
        An array containing the longitude and latitude of the two points.

    extreme_type : str
        The type of extreme latitude to calculate. Must be either 'max' or 'min'.

    Returns
    -------
    float
        The maximum or minimum latitude of the great circle arc in radians.

    Raises
    ------
    ValueError
        If `extreme_type` is not 'max' or 'min'.
    """
    # Validate extreme_type
    if (extreme_type != "max") and (extreme_type != "min"):
        raise ValueError("extreme_type must be either 'max' or 'min'")

    # Extract the two points
    n1 = gca_cart[0]
    n2 = gca_cart[1]

    # Compute dot product
    dot_n1_n2 = np.dot(n1, n2)

    # Compute denominator
    denom = (n1[2] + n2[2]) * (dot_n1_n2 - 1.0)

    # Initialize latitudes
    lon_n1, lat_n1 = gca_lonlat[0]
    lon_n2, lat_n2 = gca_lonlat[1]

    # Check if denominator is zero to avoid division by zero
    if denom != 0.0:
        d_a_max = (n1[2] * dot_n1_n2 - n2[2]) / denom

        # Handle cases where d_a_max is very close to 0 or 1
        if np.isclose(d_a_max, 0.0, atol=ERROR_TOLERANCE) or np.isclose(
            d_a_max, 1.0, atol=ERROR_TOLERANCE
        ):
            d_a_max = clip_scalar(d_a_max, 0.0, 1.0)

        # Check if d_a_max is within the valid range
        if (d_a_max > 0.0) and (d_a_max < 1.0):
            # Compute the intermediate point on the GCA
            node3 = (1.0 - d_a_max) * n1 + d_a_max * n2

            # Normalize the intermediate point
            x, y, z = _normalize_xyz_scalar(node3[0], node3[1], node3[2])
            node3_normalized = np.empty(3)
            node3_normalized[0] = x
            node3_normalized[1] = y
            node3_normalized[2] = z

            # Compute latitude of the intermediate point
            d_lat_rad = math.asin(clip_scalar(node3_normalized[2], -1.0, 1.0))

            # Return the extreme latitude
            if extreme_type == "max":
                return max3(d_lat_rad, lat_n1, lat_n2)
            else:
                return min3(d_lat_rad, lat_n1, lat_n2)

    # If denom is zero or d_a_max is not in (0,1), return max or min of lat_n1 and lat_n2
    if extreme_type == "max":
        return max(lat_n1, lat_n2)
    else:
        return min(lat_n1, lat_n2)


@njit(cache=True)
def extreme_gca_z(gca_cart, extreme_type):
    """
    Calculate the maximum or minimum latitude of a great circle arc defined
    by two 3D points.

    Parameters
    ----------
    gca_cart : numpy.ndarray
        An array containing two 3D vectors that define a great circle arc.

    extreme_type : str
        The type of extreme latitude to calculate. Must be either 'max' or 'min'.

    Returns
    -------
    float
        The maximum or minimum z of the great circle arc

    Raises
    ------
    ValueError
        If `extreme_type` is not 'max' or 'min'.
    """

    # Validate extreme_type
    if (extreme_type != "max") and (extreme_type != "min"):
        raise ValueError("extreme_type must be either 'max' or 'min'")

    # Extract the two points
    n1 = gca_cart[0]
    n2 = gca_cart[1]

    # Compute dot product
    dot_n1_n2 = np.dot(n1, n2)

    # Compute denominator
    denom = (n1[2] + n2[2]) * (dot_n1_n2 - 1.0)

    # (z) coordinate
    z_n1 = gca_cart[0][2]
    z_n2 = gca_cart[1][2]

    # Check if the denominator is zero to avoid division by zero
    if denom != 0.0:
        d_a_max = (n1[2] * dot_n1_n2 - n2[2]) / denom

        # Handle cases where d_a_max is very close to 0 or 1
        if np.isclose(d_a_max, 0.0, atol=ERROR_TOLERANCE) or np.isclose(
            d_a_max, 1.0, atol=ERROR_TOLERANCE
        ):
            d_a_max = clip_scalar(d_a_max, 0.0, 1.0)

        # Check if d_a_max is within the valid range
        if (d_a_max > 0.0) and (d_a_max < 1.0):
            # Compute the intermediate point on the GCA
            node3 = (1.0 - d_a_max) * n1 + d_a_max * n2

            # Normalize the intermediate point
            x, y, z = _normalize_xyz_scalar(node3[0], node3[1], node3[2])
            node3_normalized = np.empty(3)
            node3_normalized[0] = x
            node3_normalized[1] = y
            node3_normalized[2] = z

            d_z = clip_scalar(node3_normalized[2], -1.0, 1.0)

            if extreme_type == "max":
                return max3(d_z, z_n1, z_n2)
            else:
                return min3(d_z, z_n1, z_n2)

    # If denom is zero or d_a_max is not in (0,1), return max or min of lat_n1 and lat_n2
    if extreme_type == "max":
        return max(z_n1, z_n2)
    else:
        return min(z_n1, z_n2)


@njit(cache=True)
def compute_arc_length(pt_a, pt_b):
    """
    Compute the great circle arc length between two points on a unit sphere at constant latitude.

    Parameters
    ----------
    pt_a : tuple or array-like
        First point coordinates (x, y, z) on unit sphere
    pt_b : tuple or array-like
        Second point coordinates (x, y, z) on unit sphere

    Returns
    -------
    float
        Arc length between the points at their constant latitude
    """
    x1, y1, z1 = pt_a
    x2, y2, z2 = pt_b
    rho = np.sqrt(1.0 - z1 * z2)
    cross_2d = x1 * y2 - y1 * x2
    dot_2d = x1 * x2 + y1 * y2
    delta_theta = np.arctan2(cross_2d, dot_2d)

    return rho * abs(delta_theta)
