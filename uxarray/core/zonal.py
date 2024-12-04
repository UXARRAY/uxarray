import numpy as np
import dask.array as da


from uxarray.grid.integrate import _get_zonal_faces_weight_at_constLat


def _compute_non_conservative_zonal_mean(uxda, latitudes):
    uxgrid = uxda.uxgrid
    shape = uxda.shape[:-1] + (len(latitudes),)
    if isinstance(uxda.data, da.Array):
        # Create a Dask array for storing results
        result = da.zeros(shape, dtype=uxda.dtype)
    else:
        # Create a NumPy array for storing results
        result = np.zeros(shape, dtype=uxda.dtype)

    # TODO ---
    faces_edge_nodes_xyz = uxgrid.face_edge_nodes_xyz.values

    # Obtain computed bounds
    bounds = uxgrid.bounds.values

    for i, lat in enumerate(latitudes):
        face_indices = uxda.uxgrid.get_faces_at_constant_latitude(lat)

        print(len(face_indices))

        # z-coordinate in cartesian form
        z = np.sin(np.deg2rad(lat))

        faces_edge_nodes_xyz_candidate = faces_edge_nodes_xyz[face_indices]

        bounds_candidate = bounds[face_indices]

        weights = _get_zonal_faces_weight_at_constLat(
            faces_edge_nodes_xyz_candidate, z, bounds_candidate
        )["weight"].values

        total_weight = weights.sum()
        result[..., i] = ((uxda.data[..., face_indices] * weights) / total_weight).sum(
            axis=-1
        )

    return result
