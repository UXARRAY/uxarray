import numpy as np

# from uxarray.grid.coordinates import node_xyz_to_lonlat_rad, normalize_in_place

from uxarray.grid.coordinates import _xyz_to_lonlat_rad, _normalize_xyz
from uxarray.constants import ERROR_TOLERANCE


def _to_list(obj):
    if not isinstance(obj, list):
        if isinstance(obj, np.ndarray):
            # Convert the NumPy array to a list using .tolist()
            obj = obj.tolist()
        else:
            # If not a list or NumPy array, return the object as-is
            obj = [obj]
    return obj


def point_within_gca(pt, gca_cart, is_directed=False):
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
    # Convert the cartesian coordinates to lonlat coordinates
    pt_lonlat = np.array(_xyz_to_lonlat_rad(pt[0], pt[1], pt[2]))
    GCRv0_lonlat = np.array(
        _xyz_to_lonlat_rad(gca_cart[0][0], gca_cart[0][1], gca_cart[0][2])
    )
    GCRv1_lonlat = np.array(
        _xyz_to_lonlat_rad(gca_cart[1][0], gca_cart[1][1], gca_cart[1][2])
    )

    # Convert the list to np.float64
    gca_cart[0] = np.array(gca_cart[0], dtype=np.float64)
    gca_cart[1] = np.array(gca_cart[1], dtype=np.float64)

    # First if the input GCR is exactly 180 degree, we throw an exception, since this GCR can have multiple planes
    angle = _angle_of_2_vectors(gca_cart[0], gca_cart[1])
    if np.allclose(angle, np.pi, rtol=0, atol=ERROR_TOLERANCE):
        raise ValueError(
            "The input Great Circle Arc is exactly 180 degree, this Great Circle Arc can have multiple planes. "
            "Consider breaking the Great Circle Arc"
            "into two Great Circle Arcs"
        )

    if not np.allclose(
        np.dot(np.cross(gca_cart[0], gca_cart[1]), pt), 0, rtol=0, atol=ERROR_TOLERANCE
    ):
        return False

    if np.isclose(GCRv0_lonlat[0], GCRv1_lonlat[0], rtol=0, atol=ERROR_TOLERANCE):
        # If the pt and the GCA are on the same longitude (the y coordinates are the same)
        if np.isclose(GCRv0_lonlat[0], pt_lonlat[0], rtol=0, atol=ERROR_TOLERANCE):
            # Now use the latitude to determine if the pt falls between the interval
            return in_between(GCRv0_lonlat[1], pt_lonlat[1], GCRv1_lonlat[1])
        else:
            # If the pt and the GCA are not on the same longitude when the GCA is a longnitude arc, then the pt is not on the GCA
            return False

    # If the longnitude span is exactly 180 degree, then the GCA goes through the pole point
    if np.isclose(
        abs(GCRv1_lonlat[0] - GCRv0_lonlat[0]), np.pi, rtol=0, atol=ERROR_TOLERANCE
    ):
        # Special case, if the pt is on the pole point, then set its longitude to the GCRv0_lonlat[0]
        if np.isclose(abs(pt_lonlat[1]), np.pi / 2, rtol=0, atol=ERROR_TOLERANCE):
            pt_lonlat[0] = GCRv0_lonlat[0]
        if not np.isclose(
            GCRv0_lonlat[0], pt_lonlat[0], rtol=0, atol=ERROR_TOLERANCE
        ) and not np.isclose(
            GCRv1_lonlat[0], pt_lonlat[0], rtol=0, atol=ERROR_TOLERANCE
        ):
            return False
        else:
            # Determine the pole latitude and latitude extension
            if (GCRv0_lonlat[1] > 0 and GCRv1_lonlat[1] > 0) or (
                GCRv0_lonlat[1] < 0 and GCRv1_lonlat[1] < 0
            ):
                pole_lat = np.pi / 2 if GCRv0_lonlat[1] > 0 else -np.pi / 2
                lat_extend = (
                    abs(np.pi / 2 - abs(GCRv0_lonlat[1]))
                    + np.pi / 2
                    + abs(GCRv1_lonlat[1])
                )
            else:
                pole_lat = _decide_pole_latitude(GCRv0_lonlat[1], GCRv1_lonlat[1])
                lat_extend = (
                    abs(np.pi / 2 - abs(GCRv0_lonlat[1]))
                    + np.pi / 2
                    + abs(GCRv1_lonlat[1])
                )

            if is_directed and lat_extend >= np.pi:
                raise ValueError(
                    "The input GCA spans more than 180 degrees. Please divide it into two."
                )

            return in_between(GCRv0_lonlat[1], pt_lonlat[1], pole_lat) or in_between(
                pole_lat, pt_lonlat[1], GCRv1_lonlat[1]
            )

    if is_directed:
        # The anti-meridian case Sufficient condition: absolute difference between the longitudes of the two
        # vertices is greater than 180 degrees (π radians): abs(GCRv1_lon - GCRv0_lon) > π
        if abs(GCRv1_lonlat[0] - GCRv0_lonlat[0]) > np.pi:
            # The necessary condition: the pt longitude is on the opposite side of the anti-meridian
            # Case 1: where 0 --> x0--> 180 -->x1 -->0 case is lager than the 180degrees (pi radians)
            if GCRv0_lonlat[0] <= np.pi <= GCRv1_lonlat[0]:
                raise ValueError(
                    "The input Great Circle Arc span is larger than 180 degree, please break it into two."
                )

            # The necessary condition: the pt longitude is on the opposite side of the anti-meridian
            # Case 2: The anti-meridian case where 180 -->x0 --> 0 lon --> x1 --> 180 lon
            elif 2 * np.pi > GCRv0_lonlat[0] > np.pi > GCRv1_lonlat[0] > 0:
                return in_between(
                    GCRv0_lonlat[0], pt_lonlat[0], 2 * np.pi
                ) or in_between(0, pt_lonlat[0], GCRv1_lonlat[0])

        # The non-anti-meridian case.
        else:
            return in_between(GCRv0_lonlat[0], pt_lonlat[0], GCRv1_lonlat[0])
    else:
        # The undirected case
        # sort the longitude
        GCRv0_lonlat_min, GCRv1_lonlat_max = sorted([GCRv0_lonlat[0], GCRv1_lonlat[0]])
        if np.pi > GCRv1_lonlat_max - GCRv0_lonlat_min >= 0.0:
            return in_between(GCRv0_lonlat[0], pt_lonlat[0], GCRv1_lonlat[0])
        else:
            return in_between(GCRv1_lonlat_max, pt_lonlat[0], 2 * np.pi) or in_between(
                0.0, pt_lonlat[0], GCRv0_lonlat_min
            )


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


def _angle_of_2_vectors(u, v):
    """Calculate the angle between two 3D vectors u and v in radians. Can be
    used to calcualte the span of a GCR.

    Parameters
    ----------
    u : numpy.ndarray (float)
        The first 3D vector.
    v : numpy.ndarray (float)
        The second 3D vector.

    Returns
    -------
    float
        The angle between u and v in radians.
    """
    v_norm_times_u = np.linalg.norm(v) * u
    u_norm_times_v = np.linalg.norm(u) * v
    vec_minus = v_norm_times_u - u_norm_times_v
    vec_sum = v_norm_times_u + u_norm_times_v
    angle_u_v_rad = 2 * np.arctan2(np.linalg.norm(vec_minus), np.linalg.norm(vec_sum))
    return angle_u_v_rad


def extreme_gca_latitude(gca_cart, extreme_type):
    """Calculate the maximum or minimum latitude of a great circle arc defined
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
        The maximum or minimum latitude of the great circle arc in radians.

    Raises
    ------
    ValueError
        If `extreme_type` is not 'max' or 'min'.
    """
    extreme_type = extreme_type.lower()

    if extreme_type not in ("max", "min"):
        raise ValueError("extreme_type must be either 'max' or 'min'")

    n1, n2 = gca_cart
    dot_n1_n2 = np.dot(n1, n2)
    denom = (n1[2] + n2[2]) * (dot_n1_n2 - 1.0)
    d_a_max = (n1[2] * dot_n1_n2 - n2[2]) / denom

    d_a_max = (
        np.clip(d_a_max, 0, 1)
        if np.isclose(d_a_max, [0, 1], atol=ERROR_TOLERANCE).any()
        else d_a_max
    )

    _, lat_n1 = _xyz_to_lonlat_rad(n1[0], n1[1], n1[2])
    _, lat_n2 = _xyz_to_lonlat_rad(n2[0], n2[1], n2[2])

    if 0 < d_a_max < 1:
        node3 = (1 - d_a_max) * n1 + d_a_max * n2
        node3 = np.array(_normalize_xyz(node3[0], node3[1], node3[2]))
        d_lat_rad = np.arcsin(np.clip(node3[2], -1, 1))

        return (
            max(d_lat_rad, lat_n1, lat_n2)
            if extreme_type == "max"
            else min(d_lat_rad, lat_n1, lat_n2)
        )
    else:
        return max(lat_n1, lat_n2) if extreme_type == "max" else min(lat_n1, lat_n2)
