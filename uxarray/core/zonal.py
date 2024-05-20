import numpy as np
from uxarray.grid.integrate import _get_zonal_faces_weight_at_constLat

def _get_candidate_faces_at_constant_latitude(bounds, constLat: float) -> np.ndarray:
    """Return the indices of the faces whose latitude bounds contain the
    constant latitude.

    Parameters
    ----------
    bounds : xr.DataArray
        The latitude bounds of the faces. Expected shape is (n_face, 2).

    constLat : float
        The constant latitude to check against.

    Returns
    -------
    np.ndarray
        An array of indices of the faces whose latitude bounds contain the constant latitude.
    """

    # Extract the latitude bounds
    lat_bounds_min = bounds[:, 0, 0]  # Minimum latitude bound
    lat_bounds_max = bounds[:, 0, 1]  # Maximum latitude bound

    # Check if the constant latitude is within the bounds of each face
    within_bounds = (lat_bounds_min <= constLat) & (lat_bounds_max >= constLat)

    # Get the indices of faces where the condition is True
    candidate_faces = np.where(within_bounds)[0]

    return candidate_faces


def _non_conservative_zonal_mean_constant_one_latitude(
    uxgrid,
    face_bounds: np.ndarray,
    face_data: np.ndarray,
    constLat: float,
    is_latlonface=False,
) -> np.ndarray:
    # Get the indices of the faces whose latitude bounds contain the constant latitude    
    candidate_faces_indices = _get_candidate_faces_at_constant_latitude(
        face_bounds, constLat
    )
    candidate_face_data = face_data[..., candidate_faces_indices]

    # TODO: Get the edge connectivity of the faces
       

    # TODO: Get the edge nodes of the candidate faces
    # faces_edges_cart = # np.ndarray of dim (n_faces, n_edges, 2, 3)

    # TODO: Call the function that calculates the weights for these faces
    # Coordinates conversion: node_xyz to node_lonlat
    weight_df = _get_zonal_faces_weight_at_constLat(
        # np.array([face_0_edge_nodes, face_1_edge_nodes, face_2_edge_nodes]),
        faces_edges_cart,
        np.sin(np.deg2rad(constLat)), # Latitude in cartesian coordinates
        face_bounds,
        is_directed=False,
        is_latlonface=is_latlonface,
    )

    # Merge weights with face data
    weights = weight_df['weight'].values
    zonal_mean = (candidate_face_data * weights).sum()

    return zonal_mean


def _non_conservative_zonal_mean_constant_latitudes(
    faces_lonlat: np.ndarray,
    face_bounds: np.ndarray,
    face_data: np.ndarray,
    step_size: float,
    is_latlonface: bool = False,
) -> np.ndarray:
    # consider that the data being fed into this function may have multiple non-grid dimensions
    # (i.e. (time, level, n_face)
    # this shouldn't lead to any issues, but need to make sure that once you get to the point
    # where you obtain the indices of the faces you will be using for the zonal average, the indexing
    # is done properly along the final dimensions
    # i.e. data[..., face_indicies_at_constant_lat]

    # TODO: Loop the step size of data and calculate the zonal mean for each latitude, utilize the numpy/pandas API to do this efficiently
    latitudes = np.arange(-90, 90, step_size)
    zonal_mean = np.array([])

    for constLat in latitudes:
        zonal_mean = np.append(
            zonal_mean,
            _non_conservative_zonal_mean_constant_one_latitude(
                faces_lonlat, face_bounds, face_data, constLat, is_latlonface
            ),
        )

    # the returned array should have the same leading dimensions and a final dimension of one, indicating the mean
    # (i.e. (time, level, 1)

    return zonal_mean
