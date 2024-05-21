import numpy as np
from uxarray.grid.integrate import _get_zonal_faces_weight_at_constLat


def _get_candidate_faces_at_constant_latitude(bounds, constLat: float) -> np.ndarray:
    """Return the indices of the faces whose latitude bounds contain the
    constant latitude.

    Parameters
    ----------
    bounds : np.ndarray, shape (n_face, 2, 2)
        The latitude and longitude bounds of the faces.

    constLat : float
        The constant latitude to check against. Expected range is [-90, 90].

    Returns
    -------
    np.ndarray, shape (n_candidate_faces, )
        An array of indices of the faces whose latitude bounds contain the constant latitude `constLat`.
    """

    # Check if the constant latitude is within the range of [-90, 90]
    if constLat < -90 or constLat > 90:
        raise ValueError("The constant latitude must be within the range of [-90, 90].")

    # Extract the latitude bounds
    lat_bounds_min = bounds[:, 0, 0]  # Minimum latitude bound
    lat_bounds_max = bounds[:, 0, 1]  # Maximum latitude bound

    # Check if the constant latitude is within the bounds of each face
    within_bounds = (lat_bounds_min <= constLat) & (lat_bounds_max >= constLat)

    # Get the indices of faces where the condition is True
    candidate_faces = np.where(within_bounds)[0]

    return candidate_faces


def _non_conservative_zonal_mean_constant_one_latitude(
    face_edges_cart: np.ndarray,
    face_bounds: np.ndarray,
    face_data: np.ndarray,
    constLat: float,
    is_latlonface=False,
) -> float:
    """Helper function for _non_conservative_zonal_mean_constant_latitudes.
    Calculate the zonal mean of the data at a constant latitude.

    Parameters
    ----------
    face_edges_cart : np.ndarray, shape (n_face, n_edge, 2, 3)
        The Cartesian coordinates of the face edges.
    bounds : np.ndarray, shape (n_face, 2, 2)
        The latitude and longitude bounds of the faces.
    face_data : np.ndarray, shape (..., n_face)
        The data on the faces.
    constLat : float
        The constant latitude in degrees. Expected range is [-90, 90].
    is_latlonface : bool, optional
        A flag indicating if the current face is a latitudinal/longitudinal (latlon) face,
        meaning its edges align with lines of constant latitude or longitude. If `True`,
        edges are treated as following constant latitudinal or longitudinal lines. If `False`,
        edges are considered as great circle arcs (GCA). Default is `False`.

    Returns
    -------
    float
        The zonal mean of the data at the constant latitude.
    """

    # Get the indices of the faces whose latitude bounds contain the constant latitude
    candidate_faces_indices = _get_candidate_faces_at_constant_latitude(
        face_bounds, constLat
    )
    # Get the face data of the candidate faces
    candidate_face_data = face_data[..., candidate_faces_indices]

    # Get the list of face polygon represented by edges in Cartesian coordinates
    candidate_face_edges_cart = face_edges_cart[candidate_faces_indices]

    weight_df = _get_zonal_faces_weight_at_constLat(
        candidate_face_edges_cart,
        np.sin(np.deg2rad(constLat)),  # Latitude in cartesian coordinates
        face_bounds,
        is_directed=False,
        is_latlonface=is_latlonface,
    )

    # Compute the zonal mean(weighted average) of the candidate faces
    weights = weight_df["weight"].values
    zonal_mean = np.sum(candidate_face_data * weights) / np.sum(weights)

    return zonal_mean


def _non_conservative_zonal_mean_constant_latitudes(
    face_edges_cart: np.ndarray,
    face_bounds: np.ndarray,
    face_data: np.ndarray,
    step_size: float,
    is_latlonface=False,
) -> np.ndarray:
    """Calculate the zonal mean of the data from -90 to 90 degrees latitude,
    with a given step size.

    Parameters
    ----------
    face_edges_cart : np.ndarray, shape (n_face, n_edge, 2, 3)
        The Cartesian coordinates of the face edges.
    bounds : np.ndarray, shape (n_face, 2, 2)
        The latitude and longitude bounds of the faces.
    face_data : np.ndarray, shape (..., n_face)
        The data on the faces. It may have multiple non-grid dimensions (e.g., time, level).
    step_size : float
        The step size in degrees for the latitude.
    is_latlonface : bool, optional
        A flag indicating if the current face is a latitudinal/longitudinal (latlon) face,
        meaning its edges align with lines of constant latitude or longitude. If `True`,
        edges are treated as following constant latitudinal or longitudinal lines. If `False`,
        edges are considered as great circle arcs (GCA). Default is `False`.

    Returns
    -------
    np.ndarray
        The zonal mean of the data from -90 to 90 degrees latitude. The shape of the output
        is (..., n_latitudes), where n_latitudes is the number of latitude steps from -90 to 90.
    """

    # Generate latitudes from -90 to 90 with the given step size
    latitudes = np.arange(-90, 90 + step_size, step_size)

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

    return zonal_means
