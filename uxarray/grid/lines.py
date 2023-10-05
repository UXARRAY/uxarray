import numpy as np

from uxarray.grid.coordinates import node_xyz_to_lonlat_rad

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


def point_within_gca(pt, gca_cart):
    """Check if a point lies on a given Great Circle Arc (GCA). The anti-
    meridian case is also considered.

    Parameters
    ----------
    pt : numpy.ndarray (float)
        Cartesian coordinates of the point.
    gca_cart : numpy.ndarray of shape (2, 3), (np.float or gmpy2.mpfr)
        Cartesian coordinates of the Great Circle Arc (GCR).

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
    pt_lonlat = node_xyz_to_lonlat_rad(_to_list(pt))
    GCRv0_lonlat = node_xyz_to_lonlat_rad(_to_list(gca_cart[0]))
    GCRv1_lonlat = node_xyz_to_lonlat_rad(_to_list(gca_cart[1]))

    # Convert the list to np.float64
    gca_cart[0] = np.array(gca_cart[0], dtype=np.float64)
    gca_cart[1] = np.array(gca_cart[1], dtype=np.float64)

    # First if the input GCR is exactly 180 degree, we throw an exception, since this GCR can have multiple planes
    angle = _angle_of_2_vectors(gca_cart[0], gca_cart[1])
    if np.allclose(angle, np.pi, rtol=0, atol=ERROR_TOLERANCE):
        raise ValueError(
            "The input Great Circle Arc is exactly 180 degree, this Great Circle Arc can have multiple planes. "
            "Consider breaking the Great Circle Arc"
            "into two Great Circle Arcs")

    if not np.allclose(np.dot(np.cross(gca_cart[0], gca_cart[1]), pt),
                       0,
                       rtol=0,
                       atol=ERROR_TOLERANCE):
        return False

    # Special case: If the gcr goes across the anti-meridian
    # If the pt and the GCR are on the same longitude (the y coordinates are the same)
    if GCRv0_lonlat[0] == GCRv1_lonlat[0] == pt_lonlat[0]:
        # Now use the latitude to determine if the pt falls between the interval
        return in_between(GCRv0_lonlat[1], pt_lonlat[1], GCRv1_lonlat[1])

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
            return in_between(GCRv0_lonlat[0],
                              pt_lonlat[0], 2 * np.pi) or in_between(
                                  0, pt_lonlat[0], GCRv1_lonlat[0])

    # The non-anti-meridian case.
    else:
        return in_between(GCRv0_lonlat[0], pt_lonlat[0], GCRv1_lonlat[0])


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
    angle_u_v_rad = 2 * np.arctan2(np.linalg.norm(vec_minus),
                                   np.linalg.norm(vec_sum))
    return angle_u_v_rad
