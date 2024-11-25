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

    pt_lonlat = np.array(
        _xyz_to_lonlat_rad_scalar(pt_xyz[0], pt_xyz[1], pt_xyz[2], normalize=False)
    )
    gca_a_xyz = gca_xyz[0]

    gca_a_lonlat = np.array(
        _xyz_to_lonlat_rad_scalar(
            gca_xyz[0][0], gca_xyz[0][1], gca_xyz[0][2], normalize=False
        )
    )
    gca_b_xyz = gca_xyz[1]

    gca_b_lonlat = np.array(
        _xyz_to_lonlat_rad_scalar(
            gca_xyz[1][0], gca_xyz[1][1], gca_xyz[1][2], normalize=False
        )
    )

    return point_within_gca(
        pt_xyz,
        pt_lonlat,
        gca_a_xyz,
        gca_a_lonlat,
        gca_b_xyz,
        gca_b_lonlat,
        is_directed=is_directed,
    )


@njit(cache=True)
def point_within_gca(
    pt_xyz,
    pt_lonlat,
    gca_a_xyz,
    gca_a_lonlat,
    gca_b_xyz,
    gca_b_lonlat,
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
    if np.isclose(
        gca_a_lonlat[0], gca_b_lonlat[0], rtol=MACHINE_EPSILON, atol=MACHINE_EPSILON
    ):
        # If the pt and the GCA are on the same longitude (the y coordinates are the same)
        if np.isclose(
            gca_a_lonlat[0], pt_lonlat[0], rtol=MACHINE_EPSILON, atol=MACHINE_EPSILON
        ):
            # Now use the latitude to determine if the pt falls between the interval
            return in_between(gca_a_lonlat[1], pt_lonlat[1], gca_b_lonlat[1])
        else:
            # If the pt and the GCA are not on the same longitude when the GCA is a longitude arc, then the pt is not on the GCA
            return False
    # ==================================================================================================================
    # If the longitude span is exactly 180 degree, then the GCA goes through the pole point
    # Or if one of the endpoints is on the pole point, then the GCA goes through the pole point
    if (
        np.isclose(
            np.abs(gca_b_lonlat[0] - gca_a_lonlat[0]),
            np.pi,
            rtol=0.0,
            atol=MACHINE_EPSILON,
        )
        or np.isclose(
            np.abs(gca_a_lonlat[1]),
            np.pi / 2,
            rtol=ERROR_TOLERANCE,
            atol=ERROR_TOLERANCE,
        )
        or np.isclose(
            np.abs(gca_b_lonlat[1]),
            np.pi / 2,
            rtol=ERROR_TOLERANCE,
            atol=ERROR_TOLERANCE,
        )
    ):
        # ==============================================================================================================
        # Special case, if the pt is on the pole point, then set its longitude to the GCRv0_lonlat[0]
        # Since the point is our calculated properly, we use the atol=ERROR_TOLERANCE and rtol=ERROR_TOLERANCE
        if np.isclose(
            np.abs(pt_lonlat[1]), np.pi / 2, rtol=ERROR_TOLERANCE, atol=ERROR_TOLERANCE
        ):
            pt_lonlat[0] = gca_a_lonlat[0]

        # ==============================================================================================================
        # Special case, if one of the GCA endpoints is on the pole point, and another endpoint is not
        # then we need to check if the pt is on the GCA
        if np.isclose(
            abs(gca_a_lonlat[1]), np.pi / 2, rtol=ERROR_TOLERANCE, atol=0.0
        ) or np.isclose(
            abs(gca_b_lonlat[1]), np.pi / 2, rtol=ERROR_TOLERANCE, atol=0.0
        ):
            # Identify the non-pole endpoint
            non_pole_endpoint = None
            if not np.isclose(
                abs(gca_a_lonlat[1]), np.pi / 2, rtol=ERROR_TOLERANCE, atol=0.0
            ):
                non_pole_endpoint = gca_a_lonlat
            elif not np.isclose(
                abs(gca_b_lonlat[1]), np.pi / 2, rtol=ERROR_TOLERANCE, atol=0.0
            ):
                non_pole_endpoint = gca_b_lonlat

            if non_pole_endpoint is not None and not np.isclose(
                non_pole_endpoint[0], pt_lonlat[0], rtol=ERROR_TOLERANCE, atol=0.0
            ):
                return False
        # ==============================================================================================================
        if not np.isclose(
            gca_a_lonlat[0], pt_lonlat[0], rtol=ERROR_TOLERANCE, atol=0.0
        ) and not np.isclose(
            gca_b_lonlat[0], pt_lonlat[0], rtol=ERROR_TOLERANCE, atol=0.0
        ):
            return False
        else:
            # Determine the pole latitude and latitude extension
            if (gca_a_lonlat[1] > 0.0 and gca_b_lonlat[1] > 0.0) or (
                gca_a_lonlat[1] < 0.0 and gca_b_lonlat[1] < 0.0
            ):
                pole_lat = np.pi / 2 if gca_a_lonlat[1] > 0.0 else -np.pi / 2
                lat_extend = (
                    abs(np.pi / 2 - abs(gca_a_lonlat[1]))
                    + np.pi / 2
                    + abs(gca_b_lonlat[1])
                )
            else:
                pole_lat = _decide_pole_latitude(gca_a_lonlat[1], gca_b_lonlat[1])
                lat_extend = (
                    abs(np.pi / 2 - abs(gca_a_lonlat[1]))
                    + np.pi / 2
                    + abs(gca_b_lonlat[1])
                )

            if is_directed and lat_extend >= np.pi:
                raise ValueError(
                    "The input GCA spans more than 180 degrees. Please divide it into two."
                )

            return in_between(gca_a_lonlat[1], pt_lonlat[1], pole_lat) or in_between(
                pole_lat, pt_lonlat[1], gca_b_lonlat[1]
            )
        # endIf
        # ==============================================================================================================
    if is_directed:
        # The anti-meridian case Sufficient condition: absolute difference between the longitudes of the two
        # vertices is greater than 180 degrees (π radians): abs(GCRv1_lon - GCRv0_lon) > π
        if abs(gca_b_lonlat[0] - gca_a_lonlat[0]) > np.pi:
            # The necessary condition: the pt longitude is on the opposite side of the anti-meridian
            # Case 1: where 0 --> x0--> 180 -->x1 -->0 case is lager than the 180degrees (pi radians)
            if gca_a_lonlat[0] <= np.pi <= gca_b_lonlat[0]:
                raise ValueError(
                    "The input Great Circle Arc span is larger than 180 degree, please break it into two."
                )

            # The necessary condition: the pt longitude is on the opposite side of the anti-meridian
            # Case 2: The anti-meridian case where 180 -->x0 --> 0 lon --> x1 --> 180 lon
            elif 2 * np.pi > gca_a_lonlat[0] > np.pi > gca_b_lonlat[0] > 0.0:
                return in_between(
                    gca_a_lonlat[0], pt_lonlat[0], 2 * np.pi
                ) or in_between(0, pt_lonlat[0], gca_b_lonlat[0])

        # The non-anti-meridian case.
        else:
            return in_between(gca_a_lonlat[0], pt_lonlat[0], gca_b_lonlat[0])
    else:
        # The undirected case
        # sort the longitude
        GCRv0_lonlat_min, GCRv1_lonlat_max = sorted([gca_a_lonlat[0], gca_b_lonlat[0]])
        if np.pi > GCRv1_lonlat_max - GCRv0_lonlat_min >= 0.0:
            return in_between(gca_a_lonlat[0], pt_lonlat[0], gca_b_lonlat[0])
        else:
            return in_between(GCRv1_lonlat_max, pt_lonlat[0], 2 * np.pi) or in_between(
                0.0, pt_lonlat[0], GCRv0_lonlat_min
            )

    return None


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
