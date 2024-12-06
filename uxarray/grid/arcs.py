import numpy as np
import math


from uxarray.grid.coordinates import _xyz_to_lonlat_rad_scalar

from uxarray.grid.coordinates import (
    _normalize_xyz_scalar,
)

from uxarray.grid.utils import _angle_of_2_vectors

from uxarray.constants import ERROR_TOLERANCE, MACHINE_EPSILON

from uxarray.utils.computing import isclose, dot

from numba import njit


def _to_list(obj):
    if not isinstance(obj, list):
        if isinstance(obj, np.ndarray):
            # Convert the NumPy array to a list using .tolist()
            obj = obj.tolist()
        else:
            # If not a list or NumPy array, return the object as-is
            obj = [obj]
    return obj


def _point_within_gca_cartesian(pt_xyz, gca_xyz, is_directed=False):
    pt_xyz = np.asarray(pt_xyz)
    gca_xyz = np.asarray(gca_xyz)

    gca_a_xyz = gca_xyz[0]

    gca_b_xyz = gca_xyz[1]


    return point_within_gca(
        pt_xyz,
        gca_a_xyz,
        gca_b_xyz,
        is_directed=is_directed,
    )

#TODO: Need to rewrite the function
# @njit(cache=True)
def point_within_gca(
    pt_xyz,
    gca_a_xyz,
    gca_b_xyz,
    is_directed=False,
):
    """Check if a point lies on a given Great Circle Arc (GCA). The anti-
    meridian case is also considered.

    Parameters
    ----------
    pt : numpy.ndarray (float)
        Cartesian coordinates of the point.
    gca_cart : numpy.ndarray of shape (2, 3), (np.float or gmpy2.mpfr)
        Cartesian coordinates of the Great Circle Arc (GCR).
    is_directed : bool, optional, default = False
        If True, the GCA is considered to be directed, which means it can only from v0-->v1. If False, the GCA is undirected,
        and we will always assume the small circle (The one less than 180 degree) side is the GCA. The default is False.
        For the case of the anti-podal case, the direction is v_0--> the pole point that on the same hemisphere as v_0-->v_1

    Returns
    -------
    bool
        True if the point lies between the two endpoints of the GCR, False otherwise.

    Raises
    ------
    ValueError
        If the input GCR spans exactly 180 degrees (π radians), as this GCR can have multiple planes.
        In such cases, consider breaking the GCR into two separate GCRs.

    ValueError
        If the input GCR spans more than 180 degrees (π radians).
        In such cases, consider breaking the GCR into two separate GCRs.

    Notes
    -----
    The function checks if the given point is on the Great Circle Arc by considering its cartesian coordinates and
    accounting for the anti-meridian case.

    The anti-meridian case occurs when the GCR crosses the anti-meridian (0 longitude).
    In this case, the function handles scenarios where the GCA spans across more than 180 degrees, requiring specific operation.

    The function relies on the `_angle_of_2_vectors` and `is_between` functions to perform the necessary calculations.

    Please ensure that the input coordinates are in radians and adhere to the ERROR_TOLERANCE value for floating-point comparisons.
    """

    # ==================================================================================================================
    # 1. If the input GCR is exactly 180 degree, we throw an exception, since this GCR can have multiple planes
    angle = _angle_of_2_vectors(gca_a_xyz, gca_b_xyz)

    if np.allclose(angle, np.pi, rtol=0.0, atol=MACHINE_EPSILON):
        raise ValueError(
            "The input Great Circle Arc is exactly 180 degree, this Great Circle Arc can have multiple planes. "
            "Consider breaking the Great Circle Arc"
            "into two Great Circle Arcs"
        )

    # 2. if the `is_directed` is True, we also throw an exception if the GCR spans more than 180 degrees
    if is_directed:
        # Raise no implementation error
        raise NotImplementedError("the `is_directed` mode for `point_within_gca` has not been implemented yet.")

    # ==================================================================================================================
    # See if the point is on the plane of the GCA, because we are dealing with floating point numbers with np.dot now
    # just using the rtol=MACHINE_EPSILON, atol=MACHINE_EPSILON, but consider using the more proper error tolerance
    # in the future

    cross_product = np.cross(gca_a_xyz, gca_b_xyz)

    if not np.allclose(
        np.dot(cross_product, pt_xyz),
        0,
        rtol=MACHINE_EPSILON,
        atol=MACHINE_EPSILON,
    ):
        return False

    # ==================================================================================================================
    # Check if the point lie within the great circle arc interval
    # Compute normal vectors

    # Convert the gca_a_xyz and gca_b_xyz to lonlat
    gca_a_lonlat = _xyz_to_lonlat_rad_scalar(gca_a_xyz[0], gca_a_xyz[1], gca_a_xyz[2])
    gca_b_lonlat = _xyz_to_lonlat_rad_scalar(gca_b_xyz[0], gca_b_xyz[1], gca_b_xyz[2])

    # convert the gca_a_lonlat and gca_b_lonlat to degree
    gca_a_lonlat = np.degrees(gca_a_lonlat)
    gca_b_lonlat = np.degrees(gca_b_lonlat)

    # convert the pt_xyz to lonlat in degree
    pt_lonlat = _xyz_to_lonlat_rad_scalar(pt_xyz[0], pt_xyz[1], pt_xyz[2])
    pt_lonlat = np.degrees(pt_lonlat)

    cos_pt_a = dot(pt_xyz, gca_a_xyz)
    cos_pt_b = dot(pt_xyz, gca_b_xyz)

    # if both angles are less than 180 degree in radian, then we can directly use the following logic
    if cos_pt_a >= 0 and cos_pt_b >= 0 :
        t_a_xyz = np.cross(pt_xyz, gca_a_xyz)
        t_b_xyz = np.cross(pt_xyz, gca_b_xyz)

        # Compute dot product of tangent vectors, if negative, the point is inside the GCA
        d = dot(t_a_xyz, t_b_xyz)

        return d <= 0
    else:
        # if any of the angle is larger than 180 degree, then the point is on wrong side of the GCA
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


def _extreme_gca_latitude_cartesian(gca_cart, extreme_type):
    # should be shape [2, 2]
    gca_lonlat = np.array(
        [
            _xyz_to_lonlat_rad_scalar(gca_cart[0, 0], gca_cart[0, 1], gca_cart[0, 2]),
            _xyz_to_lonlat_rad_scalar(gca_cart[1, 0], gca_cart[1, 1], gca_cart[1, 2]),
        ]
    )

    return extreme_gca_latitude(gca_cart, gca_lonlat, extreme_type)


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
    dot_n1_n2 = dot(n1, n2)

    # Compute denominator
    denom = (n1[2] + n2[2]) * (dot_n1_n2 - 1.0)

    # Initialize latitudes
    lon_n1, lat_n1 = gca_lonlat[0]
    lon_n2, lat_n2 = gca_lonlat[1]

    # Check if denominator is zero to avoid division by zero
    if denom != 0.0:
        d_a_max = (n1[2] * dot_n1_n2 - n2[2]) / denom

        # Handle cases where d_a_max is very close to 0 or 1
        if isclose(d_a_max, 0.0, atol=ERROR_TOLERANCE) or isclose(
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
