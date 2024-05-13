import numpy as np


def _non_conservative_zonal_mean_constant_latitude(
    data: np.ndarray, lat: float
) -> np.ndarray:
    # consider that the data being fed into this function may have multiple non-grid dimensions
    # (i.e. (time, level, n_face)
    # this shouldn't lead to any issues, but need to make sure that once you get to the point
    # where you obtain the indices of the faces you will be using for the zonal average, the indexing
    # is done properly along the final dimensions
    # i.e. data[..., face_indicies_at_constant_lat]

    # TODO: obtain indices of the faces we will be considering for the zonal mean

    # the returned array should have the same leading dimensions and a final dimension of one, indicating the mean
    # (i.e. (time, level, 1)

    return None
