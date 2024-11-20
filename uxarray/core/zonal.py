import numpy as np
import dask.array as da


def _compute_zonal_mean(
    uxda,
    latitudes,
    conservative,
):
    """TODO:"""
    shape = uxda.shape[:-1] + (len(latitudes),)
    if isinstance(uxda.data, da.Array):
        # Create a Dask array for storing results
        result = da.empty(shape, dtype=uxda.dtype)
    else:
        # Create a NumPy array for storing results
        result = np.empty(shape, dtype=uxda.dtype)

    _face_centered_zonal_mean(uxda, result, latitudes, conservative)

    return result


def _face_centered_zonal_mean(uxda, result, latitudes, conservative):
    """TODO:"""

    for i, lat in enumerate(latitudes):
        if conservative:
            face_indices = uxda.uxgrid.get_faces_at_constant_latitude(lat)
            weights = uxda.uxgrid.get_weights(
                weights="face_areas", apply_to="faces", face_indices=face_indices
            )
        else:
            face_indices, edge_indices = uxda.uxgrid.get_faces_at_constant_latitude(
                lat,
                return_edge_indices=True,
            )
            weights = uxda.uxgrid.get_weights(
                weights="edge_magnitudes",
                apply_to="faces",
                face_indices=face_indices,
                edge_indices=edge_indices,
            )
        total_weight = weights.sum()
        result[..., i] = ((uxda.data[..., face_indices] * weights) / total_weight).sum(
            axis=-1
        )
