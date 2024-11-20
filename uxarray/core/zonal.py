import numpy as np
import dask.array as da


def _compute_zonal_mean(
    uxda,
    latitudes,
):
    """Compute zonal mean of data along specified latitudes."""
    shape = uxda.shape[:-1] + (len(latitudes),)
    if isinstance(uxda.data, da.Array):
        # Create a Dask array for storing results
        result = da.zeros(shape, dtype=uxda.dtype)
    else:
        # Create a NumPy array for storing results
        result = np.zeros(shape, dtype=uxda.dtype)

    _face_centered_zonal_mean(uxda, result, latitudes)

    return result


def _face_centered_zonal_mean(uxda, result, latitudes, conservative):
    """Compute face-centered zonal means and store in provided result array."""

    for i, lat in enumerate(latitudes):
        face_indices = uxda.uxgrid.get_faces_at_constant_latitude(lat)
        weights = uxda.uxgrid.get_weights(
            weights="face_areas", apply_to="faces", face_indices=face_indices
        )

        total_weight = weights.sum()
        result[..., i] = ((uxda.data[..., face_indices] * weights) / total_weight).sum(
            axis=-1
        )
