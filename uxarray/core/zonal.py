import numpy as np
import warnings
from uxarray.grid.integrate import _get_zonal_faces_weight_at_constLat


def _get_candidate_faces_at_constant_latitude(
    bounds, constLat_rad: float
) -> np.ndarray:
    """Return the indices of the faces whose latitude bounds contain the
    constant latitude.

    Parameters
    ----------
    bounds : np.ndarray, shape (n_face, 2, 2)
        The latitude and longitude bounds of the faces in radians.

    constLat_rad : float
        The constant latitude to check against in radians . Expected range is [-np.pi, np.pi].

    Returns
    -------
    np.ndarray, shape (n_candidate_faces, )
        An array of indices of the faces whose latitude bounds contain the constant latitude `constLat`.
    """

    # Check if the constant latitude is within the range of [-90, 90]
    if constLat_rad < -np.pi or constLat_rad > np.pi:
        raise ValueError(
            "The constant latitude must be within the range of [-90, 90] degree."
        )

    # Extract the latitude bounds
    lat_bounds_min = bounds[:, 0, 0]  # Minimum latitude bound
    lat_bounds_max = bounds[:, 0, 1]  # Maximum latitude bound

    # Check if the constant latitude is within the bounds of each face
    within_bounds = (lat_bounds_min <= constLat_rad) & (lat_bounds_max >= constLat_rad)

    # Get the indices of faces where the condition is True
    candidate_faces = np.where(within_bounds)[0]

    return candidate_faces


def _non_conservative_zonal_mean_constant_one_latitude(
    face_edges_cart: np.ndarray,
    face_bounds: np.ndarray,
    face_data: np.ndarray,
    constLat_deg: float,
    is_latlonface=False,
) -> float:
    """Helper function for _non_conservative_zonal_mean_constant_latitudes.
    Calculate the zonal mean of the data at a constant latitude. And if only
    one face is found, return the data of that face.

    Parameters
    ----------
    face_edges_cart : np.ndarray, shape (n_face, n_edge, 2, 3)
        The Cartesian coordinates of the face edges.
    bounds : np.ndarray, shape (n_face, 2, 2)
        The latitude and longitude bounds of the faces in radians.
    face_data : np.ndarray, shape (..., n_face)
        The data on the faces.
    constLat_deg : float
        The constant latitude in degrees. Expected range is [-90, 90].
    is_latlonface : bool, optional
        A flag indicating if the current face is a latitudinal/longitudinal (latlon) face,
        meaning its edges align with lines of constant latitude or longitude. If `True`,
        edges are treated as following constant latitudinal or longitudinal lines. If `False`,
        edges are considered as great circle arcs (GCA). Default is `False`.

    Returns
    -------
    float
        The zonal mean of the data at the constant latitude. If there are no faces whose latitude bounds contain the constant latitude, the function will return `np.nan`.
    """

    # Get the indices of the faces whose latitude bounds contain the constant latitude
    constLat_rad = np.deg2rad(constLat_deg)
    candidate_faces_indices = _get_candidate_faces_at_constant_latitude(
        face_bounds, constLat_rad
    )

    # Check if there are no candidate faces,
    if len(candidate_faces_indices) == 0:
        # Return NaN if there are no candidate faces and raise a warning saying no candidate faces found at this latitude
        warnings.warn(
            f"No candidate faces found at the constant latitude {constLat_deg} degrees."
        )
        return np.nan

    # Get the face data of the candidate faces
    candidate_face_data = face_data[..., candidate_faces_indices]

    # Get the list of face polygon represented by edges in Cartesian coordinates
    candidate_face_edges_cart = face_edges_cart[candidate_faces_indices]
    candidate_face_bounds = face_bounds[candidate_faces_indices]

    # Hardcoded the scenario when only one face is found
    if len(candidate_faces_indices) == 1:
        return candidate_face_data[0]

    weight_df = _get_zonal_faces_weight_at_constLat(
        candidate_face_edges_cart,
        np.sin(np.deg2rad(constLat_deg)),
        candidate_face_bounds,
        is_directed=False,
        is_latlonface=is_latlonface,
    )

    # Compute the zonal mean(weighted average) of the candidate faces
    weights = weight_df["weight"].values
    zonal_mean = np.sum(candidate_face_data * weights, axis=-1) / np.sum(weights)

    return zonal_mean


def _non_conservative_zonal_mean_constant_latitudes(
    face_edges_cart: np.ndarray,
    face_bounds: np.ndarray,
    face_data: np.ndarray,
    start_lat_deg: float,
    end_lat_deg: float,
    step_size_deg: float,
    is_latlonface=False,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the zonal mean of the data from start_lat_deg to end_lat_deg
    degrees latitude, with a given step size. The range of latitudes is.

    [start_lat_deg, end_lat_deg] inclusive. The step size can be positive or
    negative. If the step size is positive and start_lat_deg > end_lat_deg, the
    function will return an empty array. Similarly, if the step size is
    negative and start_lat_deg < end_lat_deg, the function will return an empty
    array.

    Parameters
    ----------
    face_edges_cart : np.ndarray, shape (n_face, n_edge, 2, 3)
        The Cartesian coordinates of the face edges.
    bounds : np.ndarray, shape (n_face, 2, 2)
        The latitude and longitude bounds of the faces.
    face_data : np.ndarray, shape (..., n_face)
        The data on the faces. It may have multiple non-grid dimensions (e.g., time, level).
    start_lat_deg : float
        The starting latitude in degrees. Expected range is [-90, 90].
    end_lat_deg : float
        The ending latitude in degrees. Expected range is [-90, 90].
    step_size_deg : float
        The step size in degrees for the latitude.
    is_latlonface : bool, optional
        A flag indicating if the current face is a latitudinal/longitudinal (latlon) face,
        meaning its edges align with lines of constant latitude or longitude. If `True`,
        edges are treated as following constant latitudinal or longitudinal lines. If `False`,
        edges are considered as great circle arcs (GCA). Default is `False`.

    Returns
    -------
    tuple
        A tuple containing:
        - np.ndarray: The latitudes used in the range [start_lat_deg to end_lat_deg] with the given step size.
        - np.ndarray: The zonal mean of the data from start_lat_deg to end_lat_deg degrees latitude, with a step size of step_size_deg. The shape of the output array is [..., n_latitudes]
        where n_latitudes is the number of latitudes in the range [start_lat_deg, end_lat_deg] inclusive. If a latitude does not have any faces whose latitude bounds contain it, the zonal mean for that latitude will be NaN.

    Raises
    ------
    ValueError
        If the start latitude is not within the range of [-90, 90].
        If the end latitude is not within the range of [-90, 90].

    Examples
    --------
    Calculate the zonal mean of the data from -90 to 90 degrees latitude with a step size of 1 degree:
    >>> face_edges_cart = np.random.rand(6, 4, 2, 3)
    >>> face_bounds = np.random.rand(6, 2, 2)
    >>> face_data = np.random.rand(3, 6)
    >>> zonal_means = _non_conservative_zonal_mean_constant_latitudes(
    ...     face_edges_cart, face_bounds, face_data, -90, 90, 1
    ... ) # will return the zonal means for latitudes in [-90, -89, ..., 89, 90]

    Calculate the zonal mean of the data from 30 to -10 degrees latitude with a negative step size of -5 degrees:
    >>> zonal_means = _non_conservative_zonal_mean_constant_latitudes(
    ...     face_edges_cart, face_bounds, face_data, 80, -10, -5
    ... ) # will return the zonal means for latitudes in [30, 20, 10, 0, -10]
    """
    # Check if the start latitude is within the range of [-90, 90]
    if start_lat_deg < -90 or start_lat_deg > 90:
        raise ValueError("The starting latitude must be within the range of [-90, 90].")
    # Check if the end latitude is within the range of [-90, 90]
    if end_lat_deg < -90 or end_lat_deg > 90:
        raise ValueError("The ending latitude must be within the range of [-90, 90].")

    # Generate latitudes from start_lat_deg to end_lat_deg with the given step size, inclusive
    latitudes = np.arange(start_lat_deg, end_lat_deg + step_size_deg, step_size_deg)

    # Initialize an empty list to store the zonal mean for each latitude
    zonal_means = []

    # Calculate the zonal mean for each latitude
    for constLat in latitudes:
        zonal_mean = _non_conservative_zonal_mean_constant_one_latitude(
            face_edges_cart, face_bounds, face_data, constLat, is_latlonface
        )
        zonal_means.append(zonal_mean)

    # Convert the list of zonal means to a NumPy array
    zonal_means = np.array(zonal_means)

    # Reshape the zonal mean array to have the same leading dimensions as the input data
    # and an additional dimension for the latitudes
    expected_shape = face_data.shape[:-1] + (len(latitudes),)
    zonal_means = zonal_means.reshape(expected_shape)

    return latitudes, zonal_means
