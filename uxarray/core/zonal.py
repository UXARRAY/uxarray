import numpy as np
import dask.array as da


def _compute_zonal_mean(
    uxda, data_mapping, latitudes, conservative=True, method="fast"
):
    """TODO:"""
    shape = uxda.shape[:-1] + (len(latitudes),)
    if isinstance(uxda.data, da.Array):
        # Create a Dask array for storing results
        result = da.empty(shape, dtype=uxda.dtype)
    else:
        # Create a NumPy array for storing results
        result = np.empty(shape, dtype=uxda.dtype)

    if data_mapping == "n_face":
        _face_centered_zonal_mean(uxda, result, latitudes, conservative, method)
    else:
        _edge_centered_zonal_mean(uxda, result, latitudes, method)

    return result


def _face_centered_zonal_mean(uxda, result, latitudes, conservative, method):
    """TODO:"""
    for i, lat in enumerate(latitudes):
        if conservative:
            face_indices = uxda.uxgrid.get_faces_at_constant_latitude(lat, method)
            weights = uxda.uxgrid.get_weights(
                weights="face_areas", apply_to="faces", face_indices=face_indices
            )
        else:
            face_indices, edge_indices = uxda.uxgrid.get_faces_at_constant_latitude(
                lat, method, return_edge_indices=True
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


def _edge_centered_zonal_mean(uxda, result, latitudes, method):
    """TODO:"""
    weights = uxda.uxgrid.edge_magnitudes

    for i, lat in enumerate(latitudes):
        edge_indices = uxda.uxgrid.get_edges_at_constant_latitude(lat, method)
        cur_weights = weights[edge_indices]
        cur_total_weight = cur_weights.sum()
        result[..., i] = (
            (uxda.data[..., edge_indices] * cur_weights) / cur_total_weight
        ).sum(axis=-1)
