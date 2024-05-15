import numpy as np
from uxarray.grid.integrate import _get_zonal_faces_weight_at_constLat


def _get_candidate_faces_at_constant_latitude(bounds, constLat) -> np.ndarray:
    # return the indices of the faces that are within the latitude bounds
    # of the constant latitude

    #TODO: Loop over the faces and check if the latitude bounds of the face overlap with the constant latitude,
    # if they do, add the face index to the list of candidate faces, utilize the numpy/pandas API to do this efficiently
    candidate_faces = np.array([])
    return candidate_faces

def _non_conservative_zonal_mean_constant_one_latitude(faces_lonlat: np.ndarray,face_bounds: np.ndarray, face_data: np.ndarray, constLat:float,is_latlonface=False) -> np.ndarray:
    #TODO: Get the data we need to do the zonal mean for the constant latitude
    candidate_faces_indices = _get_candidate_faces_at_constant_latitude(face_bounds, constLat)
    candidate_face_data = face_data[..., candidate_faces_indices]

    #TODO: Call the function that calculates the weights for these faces

    #TODO: Read the decription of _get_zonal_faces_weight_at_constLat and see how to conver the data format in a way that it can be used by the function
    # Coordinates conversion: node_xyz to node_lonlat
    weight_df = _get_zonal_faces_weight_at_constLat(np.array([
        face_0_edge_nodes, face_1_edge_nodes, face_2_edge_nodes
    ]),
        np.sin(np.deg2rad(20)),
        latlon_bounds,
        is_directed=False, is_latlonface=is_latlonface)

    #Now just simplify times the weights with the data and sum it up
    zonal_mean = (candidate_face_data * weight_df).sum()

    return zonal_mean


def _non_conservative_zonal_mean_constant_latitudes(
    faces_lonlat:np.ndarray, face_bounds: np.ndarray, face_data: np.ndarray,step_size: float,is_latlonface: bool = False

) -> np.ndarray:
    # consider that the data being fed into this function may have multiple non-grid dimensions
    # (i.e. (time, level, n_face)
    # this shouldn't lead to any issues, but need to make sure that once you get to the point
    # where you obtain the indices of the faces you will be using for the zonal average, the indexing
    # is done properly along the final dimensions
    # i.e. data[..., face_indicies_at_constant_lat]

    #TODO: Loop the step size of data and calculate the zonal mean for each latitude, utilize the numpy/pandas API to do this efficiently
    latitudes = np.arange(-90, 90, step_size)
    zonal_mean= np.array([])

    for constLat in latitudes:
        zonal_mean = np.append(zonal_mean, _non_conservative_zonal_mean_constant_one_latitude(faces_lonlat,face_bounds, face_data, constLat,is_latlonface))


    # the returned array should have the same leading dimensions and a final dimension of one, indicating the mean
    # (i.e. (time, level, 1)

    return zonal_mean
